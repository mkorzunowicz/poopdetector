using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision;
using PoopDetector.AI.Vision.Processing;
using System.Diagnostics;
using System.Linq;

namespace PoopDetector.AI.Vision.FastVLM;

/// FastVLM 0.5B ONNX wrapper (vision_encoder + embed_tokens + decoder_model_merged).
/// Step 0: splice [pre_text_embeds] + [image_embeds] + [post_text_embeds]  -> decoder(inputs_embeds) + empty KV
/// Step N: pass ONLY the new token embedding + previous KV (no image, no full prompt) with full-length attention mask.
public sealed class FastVlmSimple : IVision, IVisualLanguageModel, IDisposable
{
    // Special IDs (from tokenizer_config.json)
    public const int EndOfTextId = 151643;   // "<|endoftext|>"
    public const int ImStartId = 151644;   // "<|im_start|>"
    public const int ImEndId = 151645;   // "<|im_end|>"
    public const int ImageId = 151646;   // "<image>"

    readonly string _encPath;
    readonly string _embPath;
    readonly string _decPath;
    readonly string _vocabPath;
    readonly string _mergesPath;

    readonly FastVlmImageProcessor _image = new(448);

    InferenceSession _enc;
    InferenceSession _emb;
    InferenceSession _dec;
    HfTokenizer _tok;

    public string Name => "FastVLM-0.5B-Simple";
    public string ModelName => "FastVLM-0.5B-Simple";
    public byte[] Model => Array.Empty<byte>();
    public InferenceSession Session => _dec;
    public Microsoft.Maui.Graphics.Size InputSize => new(448, 448);
    public FastVlmImageProcessor ImageProcessor => _image;

    public FastVlmSimple(
        string visionEncoderPath,
        string embedTokensPath,
        string decoderPath,
        string vocabPath,
        string mergesPath,
        string _ /* tokenizer.json not used here */)
    {
        _encPath = visionEncoderPath;
        _embPath = embedTokensPath;
        _decPath = decoderPath;
        _vocabPath = vocabPath;
        _mergesPath = mergesPath;
    }

    public static async Task<FastVlmSimple> CreateAsync(
        string visionEncoderPath,
        string embedTokensPath,
        string decoderPath,
        string vocabPath,
        string mergesPath,
        CancellationToken cancellationToken = default)
    {
        var m = new FastVlmSimple(visionEncoderPath, embedTokensPath, decoderPath, vocabPath, mergesPath, "");
        await m.InitializeAsync();
        return m;
    }

    public async Task InitializeAsync()
    {
        await Task.Run(() =>
        {
            var encOpt = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
#if IOS
            encOpt.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE);
#endif
            _enc = new InferenceSession(File.ReadAllBytes(_encPath), encOpt);

            var cpu = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
            _emb = new InferenceSession(File.ReadAllBytes(_embPath), cpu);
            _dec = new InferenceSession(File.ReadAllBytes(_decPath), cpu);

            _tok = new HfTokenizer(_vocabPath, _mergesPath);
        });
    }

    public async Task UpdateExecutionProviderAsync(ExecutionProviders ep)
    {
        await Task.Run(() =>
        {
            _enc?.Dispose(); _emb?.Dispose(); _dec?.Dispose();

            var encOpt = BuildOptions(ep);
            var embOpt = BuildOptions(ep);
            var decOpt = BuildOptions(ep);

            _enc = new InferenceSession(File.ReadAllBytes(_encPath), encOpt);
            _emb = new InferenceSession(File.ReadAllBytes(_embPath), embOpt);
            _dec = new InferenceSession(File.ReadAllBytes(_decPath), decOpt);
        });
    }

    static SessionOptions BuildOptions(ExecutionProviders ep)
    {
        var o = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        switch (ep)
        {
            case ExecutionProviders.NNAPI: o.AppendExecutionProvider_Nnapi(); break;
            case ExecutionProviders.CoreML: o.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE); break;
            case ExecutionProviders.OpenVINO: o.AppendExecutionProvider_OpenVINO(); break;
            case ExecutionProviders.CUDA: o.AppendExecutionProvider_CUDA(new OrtCUDAProviderOptions()); break;
            case ExecutionProviders.CPU:
            default: break;
        }
        return o;
    }

    public async Task<ImageProcessingResult> ProcessImageAsync(byte[] image)
    {
        const string defaultPrompt = "Describe this image in one sentence.";
        var text = await GenerateAsync(image, defaultPrompt, 64);
        return new ImageProcessingResult(image, caption: text);
    }

    public async Task<string> GenerateAsync(byte[] image, string prompt, int maxNewTokens = 64, CancellationToken ct = default)
    {
        return await Task.Run(() =>
        {
            // 1) Vision encoder → visual embeddings
            using var bmp = _image.PreprocessSourceImage(image);
            var pixelValues = _image.GetTensorForImage(bmp);               // float32 [1,3,448,448]
            using var encOut = _enc.Run(new[] { NamedOnnxValue.CreateFromTensor("pixel_values", pixelValues) });
            var imgEmb = encOut.First().AsTensor<float>();                 // float32 [1,V,896]
            int V = (int)imgEmb.Dimensions[1];
            int H = (int)imgEmb.Dimensions[2];                             // hidden size (896)

            // 2) ChatML prompt (exact structure) with ONE <image>, then EXPAND to V tokens
            string system = "You are a helpful visual AI assistant. Respond concisely and accurately to the user's query in one sentence.";
            string user = $"<image>{(string.IsNullOrWhiteSpace(prompt) ? "Describe the image." : prompt)}";
            var oneImage = _tok.TokenizeChatWithSingleImage(system, user, addGenerationPrompt: true);
            var expanded = _tok.ExpandSingleImageToken(oneImage, V);

            // 3) Text embeddings for the WHOLE expanded sequence
            var idsT = new DenseTensor<long>(new[] { 1, expanded.Count });
            for (int i = 0; i < expanded.Count; i++) idsT[0, i] = expanded[i];
            using var embOut = _emb.Run(new[] { NamedOnnxValue.CreateFromTensor("input_ids", idsT) });
            var textEmb = embOut.First().AsTensor<float>();                // float32 [1,T,896]
            int T = (int)textEmb.Dimensions[1];

            // 4) Masked-scatter: replace the contiguous <image> block with imgEmb (NO extra scaling)
            var (start, len) = _tok.FindImageTokenBlock(expanded);
            len = System.Math.Min(len, V);

            var combined = new DenseTensor<float>(new[] { 1, T, H });
            // copy textEmb
            var textArr = textEmb.ToArray();
            var dstSpan = combined.Buffer.Span;
            for (int i = 0; i < textArr.Length; i++) dstSpan[i] = textArr[i];
            // overwrite the image span with imgEmb
            var imgArr = imgEmb.ToArray();
            for (int i = 0; i < len; i++)
            {
                int seq = start + i;
                for (int h = 0; h < H; h++)
                    combined[0, seq, h] = imgArr[i * H + h];
            }

            // 5) First pass: full embeddings + full-length mask/positions + empty KV
            var attn0 = FullOnesMask(T);
            var pos0 = PosIds(T);

            const int LAYERS = 24, HEADS = 2, HEAD_DIM = 64;
            var inputs0 = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("inputs_embeds",  combined),
                NamedOnnxValue.CreateFromTensor("attention_mask", attn0),
                NamedOnnxValue.CreateFromTensor("position_ids",   pos0),
            };
            for (int l = 0; l < LAYERS; l++)
            {
                inputs0.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{l}.key", new DenseTensor<float>(new[] { 1, HEADS, 0, HEAD_DIM })));
                inputs0.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{l}.value", new DenseTensor<float>(new[] { 1, HEADS, 0, HEAD_DIM })));
            }

            var (logits, kv) = RunDecoder(_dec, inputs0);
            int next = ArgMaxLast(logits);
            if (next == ImEndId) return string.Empty;

            // 6) Streaming decode: IMPORTANT — attention_mask length must be (past_len + 1) each step
            var generated = new List<int>();
            int pastLen = T; // number of tokens already consumed by KV

            for (int step = 0; step < System.Math.Max(1, maxNewTokens); step++)
            {
                generated.Add(next);
                if (next == ImEndId) break;

                // embed the single next id
                var one = new DenseTensor<long>(new[] { 1, 1 }); one[0, 0] = next;
                using var emb1 = _emb.Run(new[] { NamedOnnxValue.CreateFromTensor("input_ids", one) });
                var embToken = AsDense(emb1.First().AsTensor<float>()); // [1,1,896]

                // pos id = pastLen; attention mask = ones of length (pastLen + 1)
                var pos = new DenseTensor<long>(new[] { 1, 1 }); pos[0, 0] = pastLen;
                var att = FullOnesMask(pastLen + 1);

                var stepInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("inputs_embeds",  embToken),
                    NamedOnnxValue.CreateFromTensor("attention_mask", att),
                    NamedOnnxValue.CreateFromTensor("position_ids",   pos),
                };
                for (int l = 0; l < LAYERS; l++)
                {
                    stepInputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{l}.key", kv[$"past_key_values.{l}.key"]));
                    stepInputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{l}.value", kv[$"past_key_values.{l}.value"]));
                }

                (logits, kv) = RunDecoder(_dec, stepInputs);
                next = ArgMaxStream(logits, generated, repetitionPenalty: 1.1f);
                pastLen += 1;
            }

            // 7) Decode (skip specials and byte-clean)
            var decoded = _tok.DecodeClean(generated);
            return decoded.Trim();
        }, ct);
    }

    // ---- helpers ----

    static DenseTensor<long> FullOnesMask(int n)
    {
        var t = new DenseTensor<long>(new[] { 1, n });
        for (int i = 0; i < n; i++) t[0, i] = 1;
        return t;
    }

    static DenseTensor<long> PosIds(int n)
    {
        var t = new DenseTensor<long>(new[] { 1, n });
        for (int i = 0; i < n; i++) t[0, i] = i;
        return t;
    }

    static DenseTensor<float> AsDense(Tensor<float> t)
    {
        var dims = t.Dimensions.ToArray();
        var d = new DenseTensor<float>(dims);
        var src = t.ToArray();
        var dst = d.Buffer.Span;
        for (int i = 0; i < src.Length; i++) dst[i] = src[i];
        return d;
    }

    static (Tensor<float>, Dictionary<string, Tensor<float>>) RunDecoder(InferenceSession dec, List<NamedOnnxValue> inputs)
    {
        using var res = dec.Run(inputs);
        var arr = res.ToArray();
        var logits = arr[0].AsTensor<float>();
        var kv = new Dictionary<string, Tensor<float>>(System.StringComparer.Ordinal);

        int idx = 1; int layer = 0;
        while (idx + 1 < arr.Length)
        {
            kv[$"past_key_values.{layer}.key"] = arr[idx++].AsTensor<float>();
            kv[$"past_key_values.{layer}.value"] = arr[idx++].AsTensor<float>();
            layer++;
        }
        return (logits, kv);
    }

    static int ArgMaxLast(Tensor<float> logits) // [1,T,V]
    {
        int T = (int)logits.Dimensions[1];
        int V = (int)logits.Dimensions[2];
        int best = 0; float mb = float.NegativeInfinity;
        for (int i = 0; i < V; i++)
        {
            float v = logits[0, T - 1, i];
            if (v > mb) { mb = v; best = i; }
        }
        return best;
    }

    static int ArgMaxStream(Tensor<float> logits /* [1,1,V] */, List<int> recent, float repetitionPenalty)
    {
        int V = (int)logits.Dimensions[2];
        int best = 0; float mb = float.NegativeInfinity;
        var seen = recent.Count > 64 ? recent.Skip(recent.Count - 64).ToHashSet() : recent.ToHashSet();

        for (int i = 0; i < V; i++)
        {
            float v = logits[0, 0, i];
            if (seen.Contains(i)) v = v < 0 ? v * repetitionPenalty : v / repetitionPenalty;
            if (v > mb) { mb = v; best = i; }
        }
        return best;
    }

    public void Dispose()
    {
        _enc?.Dispose();
        _emb?.Dispose();
        _dec?.Dispose();
    }
}
