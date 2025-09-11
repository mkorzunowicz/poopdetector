using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision;
using PoopDetector.AI.Vision.Processing;

namespace PoopDetector.AI.Vision.FastVLM;

/// Minimal FastVLM 0.5B (int8) wrapper.
/// Uses three ONNX sessions: vision encoder, embed_tokens, decoder.
/// Returns JSON text via ImageProcessingResult.Caption.
public sealed class FastVlm : IVision, IVisualLanguageModel
{
    readonly string _encPath;
    readonly string _embPath;
    readonly string _decPath;
    readonly string _tokenizerPath;
    readonly string _vocabPath;
    readonly string _mergesPath;
    readonly FastVlmImageProcessor _image = new(448);

    InferenceSession _enc;
    InferenceSession _emb;
    InferenceSession _dec;
    ITokenizer _tok;

    public string Name => "FastVLM-0.5B-int8";
    public string ModelName => "FastVLM-0.5B-int8";
    public byte[] Model => Array.Empty<byte>();
    public InferenceSession Session => _dec; // primary
    public FastVlm(string visionEncoderPath,
                   string embedTokensPath,
                   string decoderPath,
                   string vocabPath,
                   string mergesPath,
                   string tokenizerJsonPath)
    {
        _encPath = visionEncoderPath;
        _embPath = embedTokensPath;
        _decPath = decoderPath;
        _tokenizerPath = tokenizerJsonPath;
        _vocabPath = vocabPath;
        _mergesPath = mergesPath;
        _ = InitializeAsync();
    }

    public Microsoft.Maui.Graphics.Size InputSize => new(448, 448);
    public FastVlmImageProcessor ImageProcessor => _image;

    public async Task InitializeAsync()
    {
        await Task.Run(() =>
        {
            // EP choices: CoreML for encoder on iOS, CPU for the rest
            var encOpt = new SessionOptions();
#if IOS
            encOpt.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE);
#endif
            encOpt.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            _enc = new InferenceSession(File.ReadAllBytes(_encPath), encOpt);

            var cpu = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
            _emb = new InferenceSession(File.ReadAllBytes(_embPath), cpu);
            _dec = new InferenceSession(File.ReadAllBytes(_decPath), cpu);

            //_tok = new HfTokenizer(_tokenizerPath);
            _tok = new HfTokenizer(_vocabPath, _mergesPath);
        });
    }

    public async Task UpdateExecutionProviderAsync(ExecutionProviders executionProvider)
    {
        await Task.Run(() =>
        {
            // Recreate sessions with requested EP
            _enc?.Dispose();
            _emb?.Dispose();
            _dec?.Dispose();

            var encOpt = BuildOptions(executionProvider);
            var decOpt = BuildOptions(executionProvider);
            var embOpt = BuildOptions(executionProvider);

            _enc = new InferenceSession(File.ReadAllBytes(_encPath), encOpt);
            _emb = new InferenceSession(File.ReadAllBytes(_embPath), embOpt);
            _dec = new InferenceSession(File.ReadAllBytes(_decPath), decOpt);
        });
    }

    static SessionOptions BuildOptions(ExecutionProviders ep)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };
        switch (ep)
        {
            case ExecutionProviders.NNAPI:
                options.AppendExecutionProvider_Nnapi();
                break;
            case ExecutionProviders.CoreML:
                options.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE);
                break;
            case ExecutionProviders.OpenVINO:
                options.AppendExecutionProvider_OpenVINO();
                break;
            case ExecutionProviders.CUDA:
                options.AppendExecutionProvider_CUDA(new OrtCUDAProviderOptions());
                break;
            case ExecutionProviders.CPU:
            default:
                break; // CPU always available
        }
        return options;
    }

    public async Task<ImageProcessingResult> ProcessImageAsync(byte[] image)
    {
        // default prompt for poop detection (yes/no JSON)
        const string defaultPrompt =
            "You are a vision assistant. Answer strictly JSON with keys: feces:boolean, confidence:number, explanation:string. Question: Does this image contain visible feces?";

        string json = await GenerateAsync(image, defaultPrompt, 12);
        return new ImageProcessingResult(image, caption: json);
    }

    public async Task<string> GenerateAsync(byte[] image, string prompt, int maxNewTokens = 12, CancellationToken ct = default)
    {
        // 1) Image -> encoder
        using var bmp = _image.PreprocessSourceImage(image);
        var imgTensor = _image.GetTensorForImage(bmp);

        var encInputName = _enc.InputMetadata.Keys.First();
        using var encRes = _enc.Run(new[] { NamedOnnxValue.CreateFromTensor(encInputName, imgTensor) });
        var imgEmbVal = encRes.First();
        var imgEmbTensor = imgEmbVal.AsTensor<float>();

        // 2) Tokenize prompt and embed
        var ids = _tok.Encode(prompt);
        var inputIds = new DenseTensor<long>(new[] { 1, ids.Length });
        for (int i = 0; i < ids.Length; i++) inputIds[0, i] = ids[i];

        using var embRes = _emb.Run(new[] { NamedOnnxValue.CreateFromTensor(
            FindInput(_emb, "input_ids") ?? _emb.InputMetadata.Keys.First(), inputIds) });
        var txtEmb = embRes.First().AsTensor<float>(); // shape [1, T, D]

        // 3) Decode (greedy small loop)
        var generated = new List<int>(ids);
        string last = string.Empty;
        for (int step = 0; step < Math.Max(1, maxNewTokens); step++)
        {
            var inputs = new List<NamedOnnxValue>();

            // map inputs dynamically by name keywords
            var decInputs = _dec.InputMetadata.Keys.ToArray();
            string? imgKey = decInputs.FirstOrDefault(k => k.Contains("image", StringComparison.OrdinalIgnoreCase))
                           ?? decInputs.FirstOrDefault(k => k.Contains("emb", StringComparison.OrdinalIgnoreCase));
            string? txtKey = decInputs.FirstOrDefault(k => k.Contains("token", StringComparison.OrdinalIgnoreCase) ||
                                                           k.Contains("text", StringComparison.OrdinalIgnoreCase) ||
                                                           k.Contains("input_ids", StringComparison.OrdinalIgnoreCase));

            if (imgKey == null || txtKey == null)
                throw new InvalidOperationException("Decoder input names not recognized.");

            inputs.Add(NamedOnnxValue.CreateFromTensor(imgKey, imgEmbTensor));
            inputs.Add(NamedOnnxValue.CreateFromTensor(txtKey, txtEmb));

            using var decRes = _dec.Run(inputs);
            var logits = decRes.First().AsEnumerable<float>().ToArray();

            // assume shape [1, T, V]; take last token distribution
            int vocab = logits.Length / Math.Max(1, ids.Length);
            int offset = logits.Length - vocab;
            int next = ArgMax(logits.AsSpan(offset, vocab));
            generated.Add(next);

            // early stop on EOS
            if (next == _tok.EosId) break;

            // prepare txtEmb for next step: re-embed last token only (cheap)
            var curIds = new DenseTensor<long>(new[] { 1, 1 });
            curIds[0, 0] = next;
            using var embStep = _emb.Run(new[] { NamedOnnxValue.CreateFromTensor(
                FindInput(_emb, "input_ids") ?? _emb.InputMetadata.Keys.First(), curIds) });
            txtEmb = embStep.First().AsTensor<float>();
        }

        string text = _tok.Decode(generated.Skip(ids.Length).ToArray());
        // If model already produced JSON, return as-is; otherwise wrap into JSON schema
        string json = text.Trim().StartsWith("{") ? text.Trim() :
            System.Text.Json.JsonSerializer.Serialize(new
            {
                feces = text.Contains("yes", StringComparison.OrdinalIgnoreCase),
                confidence = 0.5,
                explanation = text.Trim(),
                model = Name,
                max_new_tokens = maxNewTokens
            });

        return json;
    }

    static string? FindInput(InferenceSession s, string contains)
        => s.InputMetadata.Keys.FirstOrDefault(k => k.IndexOf(contains, StringComparison.OrdinalIgnoreCase) >= 0);

    static int ArgMax(ReadOnlySpan<float> arr)
    {
        int idx = 0; float best = float.NegativeInfinity;
        for (int i = 0; i < arr.Length; i++)
        {
            if (arr[i] > best) { best = arr[i]; idx = i; }
        }
        return idx;
    }
}

public interface IVisualLanguageModel
{
    Task<string> GenerateAsync(byte[] image, string prompt, int maxNewTokens = 12, CancellationToken ct = default);
}
