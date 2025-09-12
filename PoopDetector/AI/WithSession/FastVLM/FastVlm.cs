using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision;
using PoopDetector.AI.Vision.Processing;
using System.Diagnostics;

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
    HfTokenizer _tok;

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
        // Try a simpler, more direct prompt format
        const string defaultPrompt = "What do you see in this image?";

        string json = await GenerateAsync(image, defaultPrompt, 20); // Increase max tokens for better response
        return new ImageProcessingResult(image, caption: json);
    }

    public Task<string> GenerateAsync(byte[] image, string prompt, int maxNewTokens = 12, CancellationToken ct = default)
    {
        return Task.Run(() =>
        {
            // 1) Image -> vision encoder
            using var bmp = _image.PreprocessSourceImage(image);
            var imgTensor = _image.GetTensorForImage(bmp);

            using var encRes = _enc.Run(new[] { NamedOnnxValue.CreateFromTensor("pixel_values", imgTensor) });
            var imgEmbTensor = encRes.First().AsTensor<float>(); // shape [1, num_patches, 896]

            // Debug: Check image embedding statistics
            var imgEmbArray = imgEmbTensor.ToArray();
            var imgMin = imgEmbArray.Min();
            var imgMax = imgEmbArray.Max();
            var imgMean = imgEmbArray.Average();
            Debug.WriteLine($"Image embeddings - Min: {imgMin:F4}, Max: {imgMax:F4}, Mean: {imgMean:F4}");

            // 2) Tokenize prompt and embed
            var ids = _tok.Encode(prompt);
            Debug.WriteLine($"Prompt tokens: [{string.Join(", ", ids)}]");
            
            var inputIds = new DenseTensor<long>(new[] { 1, ids.Length });
            for (int i = 0; i < ids.Length; i++) inputIds[0, i] = ids[i];

            using var embRes = _emb.Run(new[] { NamedOnnxValue.CreateFromTensor("input_ids", inputIds) });
            var txtEmb = embRes.First().AsTensor<float>(); // shape [1, seq_len, 896]

            // Debug: Check text embedding statistics
            var txtEmbArray = txtEmb.ToArray();
            var txtMin = txtEmbArray.Min();
            var txtMax = txtEmbArray.Max();
            var txtMean = txtEmbArray.Average();
            Debug.WriteLine($"Text embeddings - Min: {txtMin:F4}, Max: {txtMax:F4}, Mean: {txtMean:F4}");

            // CRITICAL FIX: Scale image embeddings to match text embedding range
            // Image embeddings are ~100,000x larger than text embeddings, causing overflow
            var imgAbsMax = Math.Max(Math.Abs(imgMin), Math.Abs(imgMax));
            var txtAbsMax = Math.Max(Math.Abs(txtMin), Math.Abs(txtMax));
            
            if (imgAbsMax > 0 && txtAbsMax > 0)
            {
                float scaleFactor = (float)(txtAbsMax / imgAbsMax);
                Console.WriteLine($"Scaling image embeddings by factor: {scaleFactor:F6} (from range ±{imgAbsMax:F1} to ±{txtAbsMax:F4})");
                
                // Scale image embeddings to match text embedding range
                var scaledImgEmb = new DenseTensor<float>(imgEmbTensor.Dimensions.ToArray());
                var imgData = imgEmbTensor.ToArray();
                var scaledBuffer = scaledImgEmb.Buffer;
                
                for (int i = 0; i < imgData.Length; i++)
                {
                    scaledBuffer.Span[i] = imgData[i] * scaleFactor;
                }
                
                imgEmbTensor = scaledImgEmb;
                
                // Verify scaling worked
                var verifyArray = imgEmbTensor.ToArray();
                Console.WriteLine($"Scaled image embeddings - Min: {verifyArray.Min():F4}, Max: {verifyArray.Max():F4}, Mean: {verifyArray.Average():F4}");
            }

            // 3) Keep text embeddings as-is - they're already in the right range

            // 4) Concatenate image and text embeddings
            int imgSeqLen = (int)imgEmbTensor.Dimensions[1];
            int txtSeqLen = (int)txtEmb.Dimensions[1];
            int totalSeqLen = imgSeqLen + txtSeqLen;
            int embeddingDim = (int)txtEmb.Dimensions[2];

            Debug.WriteLine($"Sequence lengths - Image: {imgSeqLen}, Text: {txtSeqLen}, Total: {totalSeqLen}");

            var combinedEmb = new DenseTensor<float>(new[] { 1, totalSeqLen, embeddingDim });
            
            // Copy image embeddings first
            for (int i = 0; i < imgSeqLen; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    combinedEmb[0, i, j] = imgEmbTensor[0, i, j];
                }
            }
            
            // Copy text embeddings after image embeddings
            for (int i = 0; i < txtSeqLen; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    combinedEmb[0, imgSeqLen + i, j] = txtEmb[0, i, j];
                }
            }

            // 5) Initialize KV cache for all 24 layers
            const int numLayers = 24;
            const int numHeads = 2;  // from tensor shape [batch_size, 2, past_sequence_length, 64]
            const int headDim = 64;
            
            var pastKeyValues = new Dictionary<string, Tensor<float>>();
            
            // Initialize empty KV cache (for first forward pass)
            for (int layer = 0; layer < numLayers; layer++)
            {
                // Empty cache for first pass: shape [1, 2, 0, 64]
                var emptyKey = new DenseTensor<float>(new[] { 1, numHeads, 0, headDim });
                var emptyValue = new DenseTensor<float>(new[] { 1, numHeads, 0, headDim });
                
                pastKeyValues[$"past_key_values.{layer}.key"] = emptyKey;
                pastKeyValues[$"past_key_values.{layer}.value"] = emptyValue;
            }

            // Create attention mask and position IDs for initial sequence
            var attentionMask = new DenseTensor<long>(new[] { 1, totalSeqLen });
            for (int i = 0; i < totalSeqLen; i++) attentionMask[0, i] = 1;

            var positionIds = new DenseTensor<long>(new[] { 1, totalSeqLen });
            for (int i = 0; i < totalSeqLen; i++) positionIds[0, i] = i;

            // 6) Generation loop
            var generated = new List<int>();
            var currentInputsEmbeds = combinedEmb;
            var currentAttentionMask = attentionMask;
            var currentPositionIds = positionIds;
            var currentPastKeyValues = pastKeyValues;

            for (int step = 0; step < Math.Max(1, maxNewTokens); step++)
            {
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("inputs_embeds", currentInputsEmbeds),
                    NamedOnnxValue.CreateFromTensor("attention_mask", currentAttentionMask),
                    NamedOnnxValue.CreateFromTensor("position_ids", currentPositionIds)
                };

                // Add all past key-value pairs
                foreach (var kvp in currentPastKeyValues)
                {
                    inputs.Add(NamedOnnxValue.CreateFromTensor(kvp.Key, kvp.Value));
                }

                using var decRes = _dec.Run(inputs);
                var outputs = decRes.ToArray();
                
                // First output should be logits
                var logits = outputs[0].AsTensor<float>();

                // Get logits for the last position: shape should be [1, seq_len, vocab_size]
                int seqLen = (int)logits.Dimensions[1];
                int vocabSize = (int)logits.Dimensions[2];
                
                // Extract logits for the last token
                var lastTokenLogits = new float[vocabSize];
                for (int i = 0; i < vocabSize; i++)
                {
                    lastTokenLogits[i] = logits[0, seqLen - 1, i];
                }
                
                // Check for NaN values in logits and provide diagnostics
                int nanCount = lastTokenLogits.Count(float.IsNaN);
                if (nanCount > 0)
                {
                    Debug.WriteLine($"WARNING: Found {nanCount} NaN values in logits at step {step}!");
                    // Replace NaN with very negative values to avoid issues
                    for (int i = 0; i < lastTokenLogits.Length; i++)
                    {
                        if (float.IsNaN(lastTokenLogits[i]))
                        {
                            lastTokenLogits[i] = -1000.0f;
                        }
                    }
                }

                // Use simple greedy decoding for now to debug the issue
                int nextToken = ArgMax(lastTokenLogits);
                Debug.WriteLine($"Step {step}: Generated token {nextToken}, logits top 5: [{string.Join(", ", lastTokenLogits.Select((v, i) => new { v, i }).OrderByDescending(x => x.v).Take(5).Select(x => $"{x.i}:{x.v:F2}"))}]");
                
                generated.Add(nextToken);

                // Early stop on EOS
                if (nextToken == _tok.EosId) 
                {
                    Debug.WriteLine("Hit EOS token, stopping generation");
                    break;
                }

                // Update KV cache from model outputs (outputs 1 onwards should be updated KV pairs)
                var newPastKeyValues = new Dictionary<string, Tensor<float>>();
                int outputIndex = 1; // Skip logits (index 0)
                
                Debug.WriteLine($"Total decoder outputs: {outputs.Length}");
                
                for (int layer = 0; layer < numLayers; layer++)
                {
                    if (outputIndex < outputs.Length)
                    {
                        var keyTensor = outputs[outputIndex].AsTensor<float>();
                        
                        // Stabilize KV cache values to prevent numerical issues
                        var stabilizedKey = StabilizeTensor(keyTensor);
                        newPastKeyValues[$"past_key_values.{layer}.key"] = stabilizedKey;
                        Debug.WriteLine($"Layer {layer} key shape: [{string.Join(",", stabilizedKey.Dimensions.ToArray())}]");
                        outputIndex++;
                    }
                    if (outputIndex < outputs.Length)
                    {
                        var valueTensor = outputs[outputIndex].AsTensor<float>();
                        
                        // Stabilize KV cache values to prevent numerical issues
                        var stabilizedValue = StabilizeTensor(valueTensor);
                        newPastKeyValues[$"past_key_values.{layer}.value"] = stabilizedValue;
                        Debug.WriteLine($"Layer {layer} value shape: [{string.Join(",", stabilizedValue.Dimensions.ToArray())}]");
                        outputIndex++;
                    }
                }

                if (newPastKeyValues.Count == 0)
                {
                    Debug.WriteLine("WARNING: No KV cache outputs found! Using previous cache.");
                    newPastKeyValues = currentPastKeyValues;
                }

                // Prepare for next iteration: embed the new token
                var newTokenIds = new DenseTensor<long>(new[] { 1, 1 });
                newTokenIds[0, 0] = nextToken;
                
                using var newTokenEmbRes = _emb.Run(new[] { NamedOnnxValue.CreateFromTensor("input_ids", newTokenIds) });
                var newTokenEmb = newTokenEmbRes.First().AsTensor<float>();
                // Don't scale - keep embeddings as-is

                // For subsequent steps, we only pass the new token embedding (not the full sequence)
                currentInputsEmbeds = ConvertToDenseTensor(newTokenEmb);

                // Update attention mask to include the new token
                int newAttentionLength = (int)currentAttentionMask.Dimensions[1] + 1;
                var newAttentionMask = new DenseTensor<long>(new[] { 1, newAttentionLength });
                for (int i = 0; i < newAttentionLength; i++) newAttentionMask[0, i] = 1;

                // Position IDs for the new token only
                var newPositionIds = new DenseTensor<long>(new[] { 1, 1 });
                newPositionIds[0, 0] = newAttentionLength - 1; // Position of the new token

                currentAttentionMask = newAttentionMask;
                currentPositionIds = newPositionIds;
                currentPastKeyValues = newPastKeyValues;
            }

            Debug.WriteLine($"Generated tokens: [{string.Join(", ", generated)}]");
            
            string text = _tok.Decode(generated.ToArray());
            Debug.WriteLine($"Decoded text: '{text}'");
            
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
        }, ct);
    }

    static int ArgMax(float[] arr)
    {
        int idx = 0; 
        float best = float.NegativeInfinity;
        for (int i = 0; i < arr.Length; i++)
        {
            if (arr[i] > best) 
            { 
                best = arr[i]; 
                idx = i; 
            }
        }
        return idx;
    }

    static int SampleNextToken(float[] logits, float temperature = 1.0f, int topK = 50)
    {
        // Apply temperature scaling
        for (int i = 0; i < logits.Length; i++)
        {
            logits[i] /= temperature;
        }

        // Simple top-k sampling
        var indexed = logits.Select((value, index) => new { value, index })
                           .OrderByDescending(x => x.value)
                           .Take(topK)
                           .ToArray();

        // Find max in top-k for greedy selection
        return indexed[0].index;
    }

    static Tensor<float> ScaleEmbeddings(Tensor<float> embeddings, float targetScale = 1.0f)
    {
        // Scale embeddings to a target range to handle mixed precision models
        var data = embeddings.ToArray();
        var absMax = data.Select(Math.Abs).Max();
        
        if (absMax == 0) return embeddings; // Avoid division by zero
        
        // Scale to target range
        float scaleFactor = targetScale / (float)absMax;
        var scaled = new DenseTensor<float>(embeddings.Dimensions.ToArray());
        var buffer = scaled.Buffer;
        
        for (int i = 0; i < data.Length; i++)
        {
            buffer.Span[i] = data[i] * scaleFactor;
        }
        
        return scaled;
    }

    static Tensor<float> NormalizeEmbeddings(Tensor<float> embeddings)
    {
        // Normalize embeddings to have zero mean and unit variance
        // This helps with mixed precision models
        var data = embeddings.ToArray();
        var mean = data.Average();
        var variance = data.Select(x => Math.Pow(x - mean, 2)).Average();
        var stdDev = Math.Sqrt(variance);
        
        if (stdDev == 0) return embeddings; // Avoid division by zero
        
        var normalized = new DenseTensor<float>(embeddings.Dimensions.ToArray());
        var normalizedData = normalized.ToArray();
        
        for (int i = 0; i < data.Length; i++)
        {
            normalizedData[i] = (float)((data[i] - mean) / stdDev);
        }
        
        // Copy back to tensor
        var buffer = normalized.Buffer;
        for (int i = 0; i < normalizedData.Length; i++)
        {
            buffer.Span[i] = normalizedData[i];
        }
        
        return normalized;
    }

    static DenseTensor<float> ConvertToDenseTensor(Tensor<float> tensor)
    {
        var dimensions = tensor.Dimensions.ToArray();
        var denseTensor = new DenseTensor<float>(dimensions);
        
        // Copy data using enumerable interface
        var sourceData = tensor.ToArray();
        var destBuffer = denseTensor.Buffer;
        
        for (int i = 0; i < sourceData.Length; i++)
        {
            destBuffer.Span[i] = sourceData[i];
        }
        
        return denseTensor;
    }
    
    static DenseTensor<float> StabilizeTensor(Tensor<float> tensor)
    {
        var dimensions = tensor.Dimensions.ToArray();
        var stabilized = new DenseTensor<float>(dimensions);
        var sourceData = tensor.ToArray();
        var destBuffer = stabilized.Buffer;
        
        const float maxValue = 50.0f;  // Reasonable range for KV cache values
        const float minValue = -50.0f;
        
        for (int i = 0; i < sourceData.Length; i++)
        {
            float value = sourceData[i];
            
            // Replace NaN and infinite values with 0
            if (float.IsNaN(value) || float.IsInfinity(value))
            {
                value = 0.0f;
            }
            // Clamp extreme values
            else if (value > maxValue)
            {
                value = maxValue;
            }
            else if (value < minValue)
            {
                value = minValue;
            }
            
            destBuffer.Span[i] = value;
        }
        
        return stabilized;
    }
}

public interface IVisualLanguageModel
{
    Task<string> GenerateAsync(byte[] image, string prompt, int maxNewTokens = 12, CancellationToken ct = default);
}
