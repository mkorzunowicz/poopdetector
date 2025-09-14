using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision;
using PoopDetector.AI.Vision.Processing;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace PoopDetector.AI.Vision.FastVLM;

/// <summary>
/// FastVLM implementation for Apple FastVLM-0.5B-ONNX
/// Consolidated from working FastVlmSimple implementation
/// </summary>
public sealed class FastVlm : IVision
{
    readonly string _encPath;
    readonly string _embPath;
    readonly string _decPath;
    readonly string _tokenizerPath;
    readonly FastVlmImageProcessor _image = new(1024); // FIXED: Match Rust implementation (1024x1024)

    InferenceSession _enc;
    InferenceSession _emb; 
    InferenceSession _dec;
    ITokenizer _tok;

    public string Name => "FastVLM-0.5B";
    public string ModelName => "FastVLM-0.5B";
    public byte[] Model => Array.Empty<byte>();
    public InferenceSession Session => _dec;
    public Microsoft.Maui.Graphics.Size InputSize => new(1024, 1024); // FIXED: Match Rust implementation
    public FastVlmImageProcessor ImageProcessor => _image;

    public FastVlm(string visionEncoderPath,
                   string embedTokensPath,
                   string decoderPath,
                   string tokenizerJsonPath)
    {
        _encPath = visionEncoderPath;
        _embPath = embedTokensPath;
        _decPath = decoderPath;
        _tokenizerPath = tokenizerJsonPath;
    }

    public static async Task<FastVlm> CreateAsync(string visionEncoderPath,
                                                 string embedTokensPath,
                                                 string decoderPath,
                                                 string tokenizerJsonPath,
                                                 CancellationToken cancellationToken = default)
    {
        var model = new FastVlm(visionEncoderPath, embedTokensPath, decoderPath, tokenizerJsonPath);
        await model.InitializeAsync();
        return model;
    }

    // IVision interface implementations
    public async Task InitializeAsync()
    {
        await Task.Run(() =>
        {
            var encOpt = new SessionOptions();
#if IOS
            encOpt.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE);
#endif
            encOpt.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            _enc = new InferenceSession(File.ReadAllBytes(_encPath), encOpt);

            var cpu = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
            _emb = new InferenceSession(File.ReadAllBytes(_embPath), cpu);
            _dec = new InferenceSession(File.ReadAllBytes(_decPath), cpu);

            // Debug: Print all model input/output metadata to understand what we're working with
            Debug.WriteLine("=== DECODER MODEL METADATA ===");
            Debug.WriteLine("Expected inputs:");
            foreach (var input in _dec.InputMetadata)
            {
                Debug.WriteLine($"  - {input.Key}: {input.Value.ElementType} {string.Join("x", input.Value.Dimensions)}");
            }
            Debug.WriteLine("Expected outputs:");
            foreach (var output in _dec.OutputMetadata)
            {
                Debug.WriteLine($"  - {output.Key}: {output.Value.ElementType} {string.Join("x", output.Value.Dimensions)}");
            }
            
            Debug.WriteLine("=== VISION ENCODER METADATA ===");
            Debug.WriteLine("Expected inputs:");
            foreach (var input in _enc.InputMetadata)
            {
                Debug.WriteLine($"  - {input.Key}: {input.Value.ElementType} {string.Join("x", input.Value.Dimensions)}");
            }
            Debug.WriteLine("Expected outputs:");
            foreach (var output in _enc.OutputMetadata)
            {
                Debug.WriteLine($"  - {output.Key}: {output.Value.ElementType} {string.Join("x", output.Value.Dimensions)}");
            }
            
            Debug.WriteLine("=== EMBED TOKENS METADATA ===");
            Debug.WriteLine("Expected inputs:");
            foreach (var input in _emb.InputMetadata)
            {
                Debug.WriteLine($"  - {input.Key}: {input.Value.ElementType} {string.Join("x", input.Value.Dimensions)}");
            }
            Debug.WriteLine("Expected outputs:");
            foreach (var output in _emb.OutputMetadata)
            {
                Debug.WriteLine($"  - {output.Key}: {output.Value.ElementType} {string.Join("x", output.Value.Dimensions)}");
            }

            // Load tokenizer - fall back to vocab/merges since JSON parsing is complex
            Debug.WriteLine("Loading tokenizer for FastVLM...");
            
            // First try to find vocab/merges files in the same directory as tokenizer.json
            var tokenizerDir = Path.GetDirectoryName(_tokenizerPath) ?? "";
            var vocabPath = Path.Combine(tokenizerDir, "vocab.json");
            var mergesPath = Path.Combine(tokenizerDir, "merges.txt");
            
            if (File.Exists(vocabPath) && File.Exists(mergesPath))
            {
                _tok = new HfTokenizer(vocabPath, mergesPath);
                Debug.WriteLine($"✅ Loaded working tokenizer from vocab: {vocabPath}, merges: {mergesPath}");
            }
            else
            {
                Debug.WriteLine($"❌ Vocab/merges files not found at {vocabPath}, {mergesPath}");
                throw new FileNotFoundException($"Required tokenizer files not found. Need either:\n1. vocab.json + merges.txt in {tokenizerDir}\n2. Properly formatted tokenizer.json (not yet implemented)");
            }
        });
    }

    public async Task UpdateExecutionProviderAsync(ExecutionProviders executionProvider)
    {
        await Task.Run(() =>
        {
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

    public async Task<ImageProcessingResult> ProcessImageAsync(byte[] image)
    {
        // Use Rust-style direct prompt
        const string defaultPrompt = "Describe what you see in this image.";
        string json = await GenerateAsync(image, defaultPrompt, 30);
        return new ImageProcessingResult(image, caption: json);
    }

    // Helper method for execution provider options
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
                break;
        }
        return options;
    }

    // Original method for backward compatibility
    public async Task<string> GenerateAsync(byte[] image, string prompt, int maxNewTokens = 300, CancellationToken ct = default)
    {
        string fullResult = "";
        await GenerateStreamAsync(image, prompt, maxNewTokens, (partialText) =>
        {
            fullResult = partialText;
        }, ct);
        return fullResult;
    }

    // New streaming method that calls onTokenGenerated for each new token
    public async Task GenerateStreamAsync(byte[] image, string prompt, int maxNewTokens = 300, 
        Action<string> onTokenGenerated = null, CancellationToken ct = default)
    {
        await Task.Run(() =>
        {
            try
            {
        Debug.WriteLine($"🚀 === DIRECT APPROACH: FastVLM with input_ids + pixel_values ===");
        Debug.WriteLine($"This matches the WebGPU implementation approach");
        Debug.WriteLine($"Using maxNewTokens: {maxNewTokens}");            // 1. Process image through vision encoder
            // Input: pixel_values (float32[s0,3,s1,s2])  
            // Output: image_features (float32[s0,((((s1-1)//64))+1)*((((s2-1)//64))+1),896])
            using var bmp = _image.PreprocessSourceImage(image);
            var imgTensor = _image.GetTensorForImage(bmp);

            using var encRes = _enc.Run(new[] { NamedOnnxValue.CreateFromTensor("pixel_values", imgTensor) });
            var imgEmbTensor = encRes.First().AsTensor<float>();

            Debug.WriteLine($"Vision encoder input shape: [{string.Join("x", imgTensor.Dimensions.ToArray())}]");
            Debug.WriteLine($"Vision encoder output shape: [{string.Join("x", imgEmbTensor.Dimensions.ToArray())}]");                // 2. Apply correct FastVLM chat template format (from Rust implementation)
                string conversation = $"<|im_start|>system\nYou are a helpful vision assistant that describes images accurately.<|im_end|>\n<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
                
                // Split on <image> and handle the IMAGE_TOKEN_INDEX = 151646 (from Rust)
                var parts = conversation.Split("<image>");
                if (parts.Length != 2)
                {
                    throw new InvalidOperationException("Prompt must contain exactly one <image> placeholder");
                }
                
                string preImageText = parts[0];
                string postImageText = parts[1];
                
                // Tokenize parts
                var preTokens = string.IsNullOrEmpty(preImageText) ? new int[0] : _tok.Encode(preImageText);
                var postTokens = _tok.Encode(postImageText);
                
                const int IMAGE_TOKEN_INDEX = 151646; // <image> token from Rust implementation
                
                // Construct token sequence
                var inputTokens = new List<int>();
                inputTokens.AddRange(preTokens);
                inputTokens.Add(IMAGE_TOKEN_INDEX);
                inputTokens.AddRange(postTokens);
                
                Debug.WriteLine($"Input tokens: [{string.Join(", ", inputTokens)}]");
                
                // Convert tokens to tensor
                var inputIdsTensor = new DenseTensor<long>(new[] { 1, inputTokens.Count });
                for (int i = 0; i < inputTokens.Count; i++)
                {
                    inputIdsTensor[0, i] = inputTokens[i];
                }

                // 3. Try to run the decoder model with both image features and input tokens
                // The decoder_model_merged should handle the projection internally
                
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                    NamedOnnxValue.CreateFromTensor("pixel_values", imgTensor)  // Pass image directly to decoder
                };

                Debug.WriteLine("Running decoder with input_ids and pixel_values (direct approach)...");
                Debug.WriteLine($"Input tensor shapes:");
                Debug.WriteLine($"- input_ids: [{string.Join("x", inputIdsTensor.Dimensions.ToArray())}]");
                Debug.WriteLine($"- pixel_values: [{string.Join("x", imgTensor.Dimensions.ToArray())}]");
                
                using var decRes = _dec.Run(inputs);
                var outputs = decRes.ToArray();
                
                Debug.WriteLine($"✓ Direct approach succeeded! Decoder outputs: {outputs.Length} tensors");
                foreach (var output in outputs)
                {
                    Debug.WriteLine($"- {output.Name}: {string.Join("x", output.AsTensor<float>().Dimensions.ToArray())}");
                }
                
                var logits = outputs[0].AsTensor<float>();
                
                // Sample from logits for the last position
                int seqLen = (int)logits.Dimensions[1];
                int vocabSize = (int)logits.Dimensions[2];
                
                Debug.WriteLine($"Direct approach - Logits shape: [1, {seqLen}, {vocabSize}]");
                
                var lastTokenLogits = new float[vocabSize];
                for (int i = 0; i < vocabSize; i++)
                {
                    lastTokenLogits[i] = logits[0, seqLen - 1, i];
                }

                // Check if image information is actually being processed
                Debug.WriteLine($"Analyzing prediction confidence...");
                var topLogits = lastTokenLogits
                    .Select((logit, idx) => new { Token = idx, Logit = logit })
                    .OrderByDescending(x => x.Logit)
                    .Take(10)
                    .ToArray();
                    
                Debug.WriteLine("Top 10 predicted tokens:");
                foreach (var item in topLogits)
                {
                    var tokenText = _tok.Decode(new[] { item.Token });
                    Debug.WriteLine($"  Token {item.Token}: '{tokenText}' (logit: {item.Logit:F3})");
                }

                // Sample next token with Rust-style temperature sampling
                var nextTokenId = SampleTemperature(lastTokenLogits, 0.8f, 100);
                string result = _tok.Decode(new int[] { nextTokenId });
                
                Debug.WriteLine($"Generated token {nextTokenId} = '{result}'");
                return result;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"❌ Direct approach failed with error: {ex.Message}");
                Debug.WriteLine($"Exception type: {ex.GetType().Name}");
                if (ex.InnerException != null)
                {
                    Debug.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }
                
                // Fallback to the original approach if decoder doesn't accept pixel_values directly
                Debug.WriteLine("🔄 Falling back to embedding-based approach...");
                return GenerateAsyncFallback(image, prompt, maxNewTokens, onTokenGenerated, ct);
            }
        }, ct);
    }

    private string GenerateAsyncFallback(byte[] image, string prompt, int maxNewTokens, Action<string> onTokenGenerated, CancellationToken ct)
    {
        try
        {
            Debug.WriteLine($"=== Fallback generation with improved approach ===");
            
            // 1. Process image through vision encoder FIRST to get actual feature dimensions
            using var bmp = _image.PreprocessSourceImage(image);
            var imgTensor = _image.GetTensorForImage(bmp);

            using var encRes = _enc.Run(new[] { NamedOnnxValue.CreateFromTensor("pixel_values", imgTensor) });
            var imgEmbTensor = encRes.First().AsTensor<float>();
            
            Debug.WriteLine($"Vision encoder output shape: [{string.Join("x", imgEmbTensor.Dimensions.ToArray())}]");
            int actualImageTokens = (int)imgEmbTensor.Dimensions[1]; // Use actual sequence length

            // CRITICAL: Verify that vision features contain meaningful data
            float minVal = float.MaxValue, maxVal = float.MinValue, sum = 0f;
            int sampleCount = Math.Min(100, (int)(imgEmbTensor.Dimensions[1] * imgEmbTensor.Dimensions[2]));
            for (int i = 0; i < sampleCount; i++)
            {
                int seqIdx = i / (int)imgEmbTensor.Dimensions[2];
                int hiddenIdx = i % (int)imgEmbTensor.Dimensions[2];
                float val = imgEmbTensor[0, seqIdx, hiddenIdx];
                minVal = Math.Min(minVal, val);
                maxVal = Math.Max(maxVal, val);
                sum += val;
            }
            float avgVal = sum / sampleCount;
            Debug.WriteLine($"Vision features stats: min={minVal:F4}, max={maxVal:F4}, avg={avgVal:F4}");
            
            if (Math.Abs(avgVal) < 1e-6f && Math.Abs(maxVal) < 1e-6f)
            {
                Debug.WriteLine("🚨 WARNING: Vision features appear to be all zeros or very small!");
            }
            else
            {
                Debug.WriteLine("✅ Vision features contain meaningful data");
            }

            // Step 1: Apply correct FastVLM chat template format (from Rust implementation)
            string conversation = $"<|im_start|>system\nYou are a helpful vision assistant that describes images accurately.<|im_end|>\n<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
            
            Debug.WriteLine($"Conversation template applied: {conversation}");
            Debug.WriteLine($"Using actual image tokens from vision encoder: {actualImageTokens}");
            
            // Step 2: Tokenize the prompt parts separately and manually insert image tokens
            // Split the conversation around the <image> placeholder
            const int IMAGE_TOKEN_INDEX = 151646; // <image> token from Rust implementation
            var promptParts = conversation.Split("<image>");
            if (promptParts.Length != 2)
            {
                Debug.WriteLine("ERROR: Expected exactly one <image> placeholder in prompt");
                return "Error: Invalid prompt format";
            }
            
            // Use ACTUAL image tokens from vision encoder, not theoretical calculation
            int numImageTokens = actualImageTokens;
            
            // Tokenize the text parts
            var preImageTokens = _tok.Encode(promptParts[0]);
            var postImageTokens = _tok.Encode(promptParts[1]);
            
            Debug.WriteLine($"Pre-image text tokens: {preImageTokens.Length}");
            Debug.WriteLine($"Post-image text tokens: {postImageTokens.Length}");
            Debug.WriteLine($"Image tokens to insert: {numImageTokens}");
            
            // Step 4: Manually construct the full token sequence
            // [pre_text_tokens] + [image_token × numImageTokens] + [post_text_tokens]
            var expandedTokens = new List<int>();
            expandedTokens.AddRange(preImageTokens);
            
            // Insert the calculated number of image tokens
            for (int i = 0; i < numImageTokens; i++)
            {
                expandedTokens.Add(IMAGE_TOKEN_INDEX);
            }
            
            expandedTokens.AddRange(postImageTokens);
            
            Debug.WriteLine($"Constructed token sequence: {expandedTokens.Count} total tokens");
            Debug.WriteLine($"Breakdown: {preImageTokens.Length} pre + {numImageTokens} image + {postImageTokens.Length} post");
            
            // Step 5: Find image token positions (should be contiguous block)
            var imageTokenPositions = new List<int>();
            for (int i = 0; i < expandedTokens.Count; i++)
            {
                if (expandedTokens[i] == IMAGE_TOKEN_INDEX)
                {
                    imageTokenPositions.Add(i);
                }
            }
            
            Debug.WriteLine($"Found {imageTokenPositions.Count} image tokens starting at position {imageTokenPositions.FirstOrDefault()}");
            
            if (imageTokenPositions.Count != numImageTokens)
            {
                Debug.WriteLine($"ERROR: Expected {numImageTokens} image tokens but found {imageTokenPositions.Count}");
                return "Error: Image token count mismatch";
            }
            
            // Split tokens around the image token (LLaVA style)
            // Step 6: Split tokens around image token positions to prepare for embedding replacement
            if (imageTokenPositions.Count == 0)
            {
                Debug.WriteLine("ERROR: No image tokens found in expanded prompt");
                return "Error: Could not find image tokens in prompt";
            }
            
            // The image tokens should be a contiguous block
            int firstImageTokenIdx = imageTokenPositions[0];
            int lastImageTokenIdx = imageTokenPositions[imageTokenPositions.Count - 1];
            
            // Verify they are contiguous
            if (lastImageTokenIdx - firstImageTokenIdx + 1 != numImageTokens)
            {
                Debug.WriteLine($"WARNING: Image tokens are not contiguous! First: {firstImageTokenIdx}, Last: {lastImageTokenIdx}");
            }
            
            var textTokensBeforeImage = expandedTokens.Take(firstImageTokenIdx).ToArray();
            var textTokensAfterImage = expandedTokens.Skip(lastImageTokenIdx + 1).ToArray();
            
            Debug.WriteLine($"Text before image ({textTokensBeforeImage.Length}): [{string.Join(", ", textTokensBeforeImage.Take(10))}...]");
            Debug.WriteLine($"Text after image ({textTokensAfterImage.Length}): [{string.Join(", ", textTokensAfterImage.Take(10))}...]");
            Debug.WriteLine($"Image tokens to replace: {numImageTokens} tokens from position {firstImageTokenIdx} to {lastImageTokenIdx}");
            
            // Get embeddings for text parts only (no image token)
            Tensor<float> preImageEmbeddings = null;
            Tensor<float> postImageEmbeddings = null;
            
            if (textTokensBeforeImage.Length > 0)
            {
                var preImageTensor = new DenseTensor<long>(new[] { 1, textTokensBeforeImage.Length });
                for (int i = 0; i < textTokensBeforeImage.Length; i++)
                {
                    preImageTensor[0, i] = textTokensBeforeImage[i];
                }
                using var preEmbRes = _emb.Run(new[] { NamedOnnxValue.CreateFromTensor("input_ids", preImageTensor) });
                preImageEmbeddings = preEmbRes.First().AsTensor<float>();
            }
            
            if (textTokensAfterImage.Length > 0)
            {
                var postImageTensor = new DenseTensor<long>(new[] { 1, textTokensAfterImage.Length });
                for (int i = 0; i < textTokensAfterImage.Length; i++)
                {
                    postImageTensor[0, i] = textTokensAfterImage[i];
                }
                using var postEmbRes = _emb.Run(new[] { NamedOnnxValue.CreateFromTensor("input_ids", postImageTensor) });
                postImageEmbeddings = postEmbRes.First().AsTensor<float>();
            }
            
            // Now combine embeddings in LLaVA style: [pre_text] + [image_features] + [post_text]
            var combinedEmbeddings = CombineEmbeddingsLlavaStyle(preImageEmbeddings, imgEmbTensor, postImageEmbeddings);
            Debug.WriteLine($"Combined embeddings shape: [{string.Join("x", combinedEmbeddings.Dimensions.ToArray())}]");

            // Use the combined embeddings directly (no separate text/image combination needed)
            var combinedFeatures = combinedEmbeddings;

            // 5. Single-step generation with ALL required inputs matching Netron specs
            int currentSeqLen = (int)combinedFeatures.Dimensions[1];
            
            // attention_mask: int64[batch_size,total_sequence_length] - matches current sequence
            var attentionMask = new DenseTensor<long>(new[] { 1, currentSeqLen });
            for (int i = 0; i < currentSeqLen; i++) attentionMask[0, i] = 1;

            // position_ids: int64[batch_size,sequence_length] - matches current sequence  
            var positionIds = new DenseTensor<long>(new[] { 1, currentSeqLen });
            for (int i = 0; i < currentSeqLen; i++) positionIds[0, i] = i;

            // Initialize proper KV cache matching exact Netron specs
            const int numLayers = 24; // FastVLM 0.5B has 24 layers (0-23)
            const int numHeads = 2;   // From Netron: tensor: float32[batch_size,2,past_sequence_length,64]
            const int headDim = 64;   // From Netron: 64 dimension
            const int pastSeqLen = 0; // Empty cache for first generation
            
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("inputs_embeds", combinedFeatures),
                NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask),
                NamedOnnxValue.CreateFromTensor("position_ids", positionIds)
            };

            // Add empty KV cache for all layers exactly matching Netron specs
            // Each layer expects: float32[batch_size,2,past_sequence_length,64]
            for (int layer = 0; layer < numLayers; layer++)
            {
                var emptyKey = new DenseTensor<float>(new[] { 1, numHeads, pastSeqLen, headDim });
                var emptyValue = new DenseTensor<float>(new[] { 1, numHeads, pastSeqLen, headDim });
                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layer}.key", emptyKey));
                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layer}.value", emptyValue));
            }

            Debug.WriteLine($"Running decoder with {inputs.Count} inputs:");
            foreach (var input in inputs)
            {
                if (input.Value is Tensor<float> floatTensor)
                {
                    Debug.WriteLine($"- {input.Name}: [{string.Join("x", floatTensor.Dimensions.ToArray())}] (float32)");
                }
                else if (input.Value is Tensor<long> longTensor)
                {
                    Debug.WriteLine($"- {input.Name}: [{string.Join("x", longTensor.Dimensions.ToArray())}] (int64)");
                }
                else
                {
                    Debug.WriteLine($"- {input.Name}: (unknown type)");
                }
            }

            using var decRes = _dec.Run(inputs);
            
            // Expected output: logits (float32[batch_size,sequence_length,151646])
            var logits = decRes.First().AsTensor<float>();

            // Get logits for the last position
            int seqLen = (int)logits.Dimensions[1];
            int vocabSize = (int)logits.Dimensions[2];
            
            Debug.WriteLine($"Decoder output logits shape: [1, {seqLen}, {vocabSize}] (expected vocab_size=151646)");
            
            var lastTokenLogits = new float[vocabSize];
            for (int i = 0; i < vocabSize; i++)
            {
                lastTokenLogits[i] = logits[0, seqLen - 1, i];
            }

            var nextTokenId = SampleTemperature(lastTokenLogits, 0.8f, 100);
            
            // Generate multiple tokens instead of just one
            var generatedTokens = new List<int> { nextTokenId };
            
            Debug.WriteLine($"Step 0: Generated token {nextTokenId} = '{_tok.Decode(new[] { nextTokenId })}'");
            
            // Call streaming callback for the first token
            if (onTokenGenerated != null)
            {
                try
                {
                    // Filter out EOS tokens for display  
                    var tokensForDisplay = generatedTokens.Where(token => !IsEosToken(token)).ToArray();
                    if (tokensForDisplay.Length > 0)
                    {
                        string firstToken = _tok.Decode(tokensForDisplay).Trim();
                        firstToken = firstToken.Replace("<|im_end|>", "").Replace("<|im_start|>", "").Trim();
                        onTokenGenerated(firstToken);
                    }
                }
                catch (Exception callbackEx)
                {
                    Debug.WriteLine($"[FastVLM] Error in streaming callback for first token: {callbackEx.Message}");
                }
            }
            
            // Check for EOS tokens after first generation
            if (IsEosToken(nextTokenId))
            {
                Debug.WriteLine($"Generated EOS token ({nextTokenId}) on first step, stopping generation");
                // Still decode what we have, excluding EOS tokens
                var cleanTokens = generatedTokens.Where(token => !IsEosToken(token)).ToArray();
                string firstResult = _tok.Decode(cleanTokens).Trim();
                firstResult = firstResult.Replace("<|im_end|>", "").Replace("<|im_start|>", "").Trim();
                Debug.WriteLine($"Final cleaned result: '{firstResult}'");
                return firstResult;
            }
            
            Debug.WriteLine($"[FastVLM] Starting generation loop with maxNewTokens={maxNewTokens}");
            
            for (int step = 1; step < maxNewTokens; step++)
            {
                try
                {
                    // For subsequent tokens, we need to extend the input embeddings
                    // Get embeddings for the newly generated token
                    var newTokenTensor = new DenseTensor<long>(new[] { 1, 1 });
                    newTokenTensor[0, 0] = nextTokenId;
                    
                    using var newTokenEmbRes = _emb.Run(new[] { NamedOnnxValue.CreateFromTensor("input_ids", newTokenTensor) });
                    var newTokenEmbedding = newTokenEmbRes.First().AsTensor<float>();
                    
                    // Extend the combined features with the new token embedding
                    combinedFeatures = ExtendEmbeddings(combinedFeatures, newTokenEmbedding);
                    
                    int newSeqLen = (int)combinedFeatures.Dimensions[1];
                    var newAttentionMask = new DenseTensor<long>(new[] { 1, newSeqLen });
                    for (int i = 0; i < newSeqLen; i++) newAttentionMask[0, i] = 1;
                    
                    var newPositionIds = new DenseTensor<long>(new[] { 1, newSeqLen });
                    for (int i = 0; i < newSeqLen; i++) newPositionIds[0, i] = i;
                    
                    var newInputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("inputs_embeds", combinedFeatures),
                        NamedOnnxValue.CreateFromTensor("attention_mask", newAttentionMask),
                        NamedOnnxValue.CreateFromTensor("position_ids", newPositionIds)
                    };
                    
                    // Add empty KV cache again (for simplicity)
                    for (int layer = 0; layer < numLayers; layer++)
                    {
                        var emptyKey = new DenseTensor<float>(new[] { 1, numHeads, pastSeqLen, headDim });
                        var emptyValue = new DenseTensor<float>(new[] { 1, numHeads, pastSeqLen, headDim });
                        newInputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layer}.key", emptyKey));
                        newInputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{layer}.value", emptyValue));
                    }
                    
                    using var newDecRes = _dec.Run(newInputs);
                    var newLogits = newDecRes.First().AsTensor<float>();
                    
                    int newSeqLenOut = (int)newLogits.Dimensions[1];
                    var newLastTokenLogits = new float[vocabSize];
                    for (int i = 0; i < vocabSize; i++)
                    {
                        newLastTokenLogits[i] = newLogits[0, newSeqLenOut - 1, i];
                    }
                    
                    nextTokenId = SampleTemperature(newLastTokenLogits, 0.8f, 100);
                    generatedTokens.Add(nextTokenId);
                    
                    string tokenText = _tok.Decode(new[] { nextTokenId });
                    
                    Debug.WriteLine($"[FastVLM] Step {step}: Generated token {nextTokenId} = '{tokenText}'");
                    
                    // Call streaming callback with current partial result
                    if (onTokenGenerated != null)
                    {
                        try
                        {
                            // Filter out EOS tokens for streaming display
                            var tokensForDisplay = generatedTokens.Where(token => !IsEosToken(token)).ToArray();
                            string currentResult = _tok.Decode(tokensForDisplay).Trim();
                            
                            // Clean up special tokens from display
                            currentResult = currentResult.Replace("<|im_end|>", "").Replace("<|im_start|>", "").Trim();
                            
                            onTokenGenerated(currentResult);
                        }
                        catch (Exception callbackEx)
                        {
                            Debug.WriteLine($"[FastVLM] Error in streaming callback: {callbackEx.Message}");
                        }
                    }
                    
                    // Check for EOS token after generating it
                    if (IsEosToken(nextTokenId))
                    {
                        Debug.WriteLine($"[FastVLM] Generated EOS token ({nextTokenId}), stopping generation");
                        break;
                    }
                    
                    Debug.WriteLine($"[FastVLM] Step {step} completed, continuing generation");
                    
                    // Let the model decide when to stop by generating EOS token
                    // No artificial stopping - trust the model's training
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[FastVLM] Exception at step {step}: {ex.Message}");
                    Debug.WriteLine($"[FastVLM] Stack trace: {ex.StackTrace}");
                    Debug.WriteLine($"[FastVLM] Breaking from generation loop due to exception");
                    break;
                }
            }
            
            Debug.WriteLine($"[FastVLM] Generation loop ended after {generatedTokens.Count} tokens");
            Debug.WriteLine($"[FastVLM] Final tokens: [{string.Join(", ", generatedTokens)}]");
            
            // Decode all generated tokens together for proper BPE handling
            Debug.WriteLine($"[FastVLM] Final token count: {generatedTokens.Count}");
            Debug.WriteLine($"[FastVLM] Generated tokens: [{string.Join(", ", generatedTokens)}]");
            
            // Filter out EOS tokens before final decoding
            var tokensForDecoding = generatedTokens.Where(token => !IsEosToken(token)).ToArray();
            string result = _tok.Decode(tokensForDecoding).Trim();
            
            // Clean up any remaining special tokens that might appear in text
            result = result.Replace("<|im_end|>", "").Replace("<|im_start|>", "").Trim();
            
            Debug.WriteLine($"[FastVLM] Final cleaned result: '{result}'");

            return result;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Error in fallback: {ex}");
            return $"Error: {ex.Message}";
        }
    }

    private string ApplyChatTemplate(string prompt)
    {
        // Simplified chat template similar to SmolVLM
        return $"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
    }

    private Tensor<float> CombineFeatures(Tensor<float> imageFeatures, Tensor<float> textFeatures)
    {
        // LLaVA-style feature combination: replace IMAGE_TOKEN_INDEX with image features
        var textShape = textFeatures.Dimensions.ToArray();
        var imgShape = imageFeatures.Dimensions.ToArray();
        
        Debug.WriteLine($"CombineFeatures: Text shape [{string.Join("x", textShape)}], Image shape [{string.Join("x", imgShape)}]");
        
        int textSeqLen = (int)textShape[1];
        int hiddenSize = (int)textShape[2];
        int imgSeqLen = (int)imgShape[1];
        
        // Find the position of IMAGE_TOKEN_INDEX (-200) in the text embeddings
        // Since embed_tokens likely maps -200 to a special embedding, we need to replace it
        
        // For simplicity, assume the image token is at a known position (after BOS if present)
        // In a real implementation, we'd scan for the special token embedding pattern
        
        // Calculate new sequence length: text tokens + image features - 1 (replacing image token)
        int newSeqLen = textSeqLen + imgSeqLen - 1; // -1 because we replace the image token
        var combined = new DenseTensor<float>(new[] { 1, newSeqLen, hiddenSize });
        
        // Strategy: assume IMAGE_TOKEN_INDEX is at position 1 (after any BOS token)
        // This follows the Apple FastVLM format where <image> is early in the sequence
        int imageTokenPos = 1;
        if (textSeqLen > imageTokenPos)
        {
            // Copy text embeddings before image position
            for (int i = 0; i < imageTokenPos; i++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    combined[0, i, j] = textFeatures[0, i, j];
                }
            }
            
            // Insert image features (projected to text embedding space)
            for (int i = 0; i < imgSeqLen; i++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    // Use image features directly (decoder should handle projection)
                    if (j < imgShape[2])
                    {
                        combined[0, imageTokenPos + i, j] = imageFeatures[0, i, j];
                    }
                    else
                    {
                        combined[0, imageTokenPos + i, j] = 0f; // Pad if needed
                    }
                }
            }
            
            // Copy remaining text embeddings after image
            int remainingTextStart = imageTokenPos + 1;
            int combinedPos = imageTokenPos + imgSeqLen;
            for (int i = remainingTextStart; i < textSeqLen; i++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    combined[0, combinedPos, j] = textFeatures[0, i, j];
                }
                combinedPos++;
            }
        }
        else
        {
            // Fallback: simple concatenation if text is too short
            Debug.WriteLine("Warning: Text sequence too short, using fallback concatenation");
            return CombineFeaturesSimple(imageFeatures, textFeatures);
        }
        
        Debug.WriteLine($"Combined features shape: [{string.Join("x", combined.Dimensions.ToArray())}]");
        return combined;
    }
    
    private Tensor<float> CombineFeaturesSimple(Tensor<float> imageFeatures, Tensor<float> textFeatures)
    {
        // Fallback: simple concatenation
        var imgShape = imageFeatures.Dimensions.ToArray();
        var textShape = textFeatures.Dimensions.ToArray();
        
        var combinedLength = imgShape[1] + textShape[1];
        var combined = new DenseTensor<float>(new[] { 1, combinedLength, imgShape[2] });
        
        // Copy image features first
        for (int i = 0; i < imgShape[1]; i++)
        {
            for (int j = 0; j < imgShape[2]; j++)
            {
                combined[0, i, j] = imageFeatures[0, i, j];
            }
        }
        
        // Copy text features
        for (int i = 0; i < textShape[1]; i++)
        {
            for (int j = 0; j < Math.Min(textShape[2], imgShape[2]); j++)
            {
                combined[0, imgShape[1] + i, j] = textFeatures[0, i, j];
            }
        }
        
        return combined;
    }

    /// <summary>
    /// Check if a token ID represents an EOS (End of Sequence) token
    /// </summary>
    private bool IsEosToken(int tokenId)
    {
        // Known EOS tokens for FastVLM:
        // 151645 = <|im_end|> (primary EOS)
        // Also check for standard EOS tokens that might be used
        var eosTokens = new[] { 151645, _tok.EosId, 2, 0 };
        bool isEos = tokenId == 151645 || // <|im_end|> 
               tokenId == _tok.EosId || // Tokenizer's defined EOS
               tokenId == 2 || // Standard </s> token
               tokenId == 0;   // Sometimes used as EOS in some models
        
        if (isEos)
        {
            Debug.WriteLine($"[FastVLM] EOS token detected: {tokenId} (known EOS tokens: [{string.Join(", ", eosTokens)}])");
        }
        
        return isEos;
    }

    private int SampleTemperature(float[] logits, float temperature, int topK)
    {
        // Apply temperature scaling
        for (int i = 0; i < logits.Length; i++)
        {
            logits[i] /= temperature;
        }

        // Convert to probabilities with softmax
        float maxLogit = logits.Max();
        var probs = new float[logits.Length];
        float sum = 0f;
        
        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = (float)Math.Exp(logits[i] - maxLogit);
            sum += probs[i];
        }
        
        for (int i = 0; i < probs.Length; i++)
        {
            probs[i] /= sum;
        }

        // Top-k sampling (matching Rust implementation)
        var topIndices = probs
            .Select((prob, idx) => new { Prob = prob, Index = idx })
            .OrderByDescending(x => x.Prob)
            .Take(topK)
            .ToArray();

        // Sample from top-k (matching Rust random sampling approach)
        var random = new Random();
        float randomValue = (float)random.NextDouble();
        float cumulative = 0f;
        
        // Renormalize top-k probabilities
        float topSum = topIndices.Sum(x => x.Prob);
        
        foreach (var item in topIndices)
        {
            cumulative += item.Prob / topSum;
            if (randomValue <= cumulative)
            {
                return item.Index;
            }
        }
        
        // Fallback to most likely token
        return topIndices[0].Index;
    }

    private int SampleNextToken(float[] logits, List<int> previousTokens = null)
    {
        // Much more restrictive token filtering - limit to basic vocabulary
        const int maxValidTokenId = 15000; // Reduced significantly from 32000
        
        // Apply repetition penalty to avoid getting stuck
        if (previousTokens != null && previousTokens.Count > 0)
        {
            const float repetitionPenalty = 2.0f; // Stronger penalty
            var recentTokens = previousTokens.TakeLast(20).ToHashSet(); // More context
            
            Debug.WriteLine($"Applying repetition penalty to {recentTokens.Count} recent tokens: [{string.Join(", ", recentTokens)}]");
            
            for (int i = 0; i < Math.Min(logits.Length, maxValidTokenId); i++)
            {
                if (recentTokens.Contains(i))
                {
                    // Stronger penalization of recent tokens
                    logits[i] = logits[i] < 0 ? logits[i] * repetitionPenalty : logits[i] / repetitionPenalty;
                }
            }
        }

        // Zero out tokens beyond vocab size and also filter out problematic high ID tokens
        for (int i = maxValidTokenId; i < logits.Length; i++)
        {
            logits[i] = float.NegativeInfinity;
        }
        
        // Also filter out tokens that are likely to be garbage (very high IDs within the range)
        for (int i = 10000; i < maxValidTokenId; i++)
        {
            // Reduce probability of very high token IDs
            if (logits[i] > 0)
                logits[i] *= 0.1f; // Reduce by 90%
        }

        // Use deterministic sampling (match Apple's do_sample: false)
        // No temperature scaling needed for deterministic output

        // Convert to probabilities using softmax
        float maxLogit = logits.Max();
        var probs = new float[logits.Length];
        float sum = 0f;
        
        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = (float)Math.Exp(logits[i] - maxLogit);
            sum += probs[i];
        }
        
        for (int i = 0; i < probs.Length; i++)
        {
            probs[i] /= sum;
        }

        // Top-k sampling but smaller k to avoid garbage tokens
        const int topK = 20;
        var topIndices = probs
            .Select((prob, idx) => new { Prob = prob, Index = idx })
            .Where(x => x.Index < maxValidTokenId) // Only valid tokens
            .OrderByDescending(x => x.Prob)
            .Take(topK)
            .ToArray();

        if (topIndices.Length == 0)
        {
            // Fallback to first valid token
            return 0;
        }

        // Sample from top-k
        var random = new Random();
        float randomValue = (float)random.NextDouble();
        float cumulative = 0f;
        
        // Renormalize top-k probabilities
        float topSum = topIndices.Sum(x => x.Prob);
        
        foreach (var item in topIndices)
        {
            cumulative += item.Prob / topSum;
            if (randomValue <= cumulative)
            {
                Debug.WriteLine($"Sampled token {item.Index} with prob {item.Prob:F6}");
                return item.Index;
            }
        }
        
        // Fallback to most likely token
        return topIndices[0].Index;
    }

    private bool IsValidToken(int tokenId, string tokenText)
    {
        // Filter out tokens that are likely to be garbage
        
        // Reject very high token IDs (likely special tokens or garbage)
        if (tokenId > 15000) return false;
        
        // Reject empty or whitespace-only tokens that might cause issues
        if (string.IsNullOrEmpty(tokenText)) return false;
        
        // Reject tokens that are just weird symbols or control characters
        if (tokenText.All(c => char.IsControl(c) || char.IsSurrogate(c))) return false;
        
        // Reject tokens that look like encoding artifacts (all caps fragments without vowels)
        if (tokenText.Length > 2 && tokenText.All(char.IsUpper) && 
            !tokenText.Any(c => "AEIOU".Contains(c)) && 
            tokenText.All(char.IsLetter))
        {
            return false;
        }
        
        return true;
    }

    /// <summary>
    /// Extends existing embeddings with a new token embedding
    /// </summary>
    private DenseTensor<float> ExtendEmbeddings(DenseTensor<float> existingEmbeddings, Tensor<float> newTokenEmbedding)
    {
        try
        {
            int existingLength = (int)existingEmbeddings.Dimensions[1];
            int hiddenSize = (int)existingEmbeddings.Dimensions[2];
            int newLength = existingLength + 1;
            
            Debug.WriteLine($"[FastVLM] ExtendEmbeddings: {existingLength} -> {newLength} tokens, hidden size: {hiddenSize}");
            
            // Safety check for very long sequences
            if (newLength > 4096)
            {
                Debug.WriteLine($"[FastVLM] Warning: Very long sequence ({newLength} tokens), this might cause memory issues");
            }
            
            var extended = new DenseTensor<float>(new[] { 1, newLength, hiddenSize });
            
            // Copy existing embeddings
            var existingArray = existingEmbeddings.ToArray();
            var newArray = newTokenEmbedding.ToArray();
            
            if (newArray.Length != hiddenSize)
            {
                Debug.WriteLine($"[FastVLM] Warning: New token embedding size mismatch. Expected {hiddenSize}, got {newArray.Length}");
            }
            
            for (int seq = 0; seq < existingLength; seq++)
            {
                for (int hidden = 0; hidden < hiddenSize; hidden++)
                {
                    extended[0, seq, hidden] = existingArray[seq * hiddenSize + hidden];
                }
            }
            
            // Add new token embedding at the end
            for (int hidden = 0; hidden < hiddenSize; hidden++)
            {
                extended[0, existingLength, hidden] = newArray[hidden];
            }
            
            Debug.WriteLine($"[FastVLM] Successfully extended embeddings from {existingLength} to {newLength} tokens");
            return extended;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[FastVLM] Error in ExtendEmbeddings: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// Extract a slice of embeddings from a tensor (helper for Rust-style fusion)
    /// </summary>
    private DenseTensor<float> ExtractEmbeddingSlice(Tensor<float> embeddings, int startPos, int length)
    {
        if (length <= 0) return null;
        
        int hiddenSize = (int)embeddings.Dimensions[2];
        var slice = new DenseTensor<float>(new[] { 1, length, hiddenSize });
        
        for (int seq = 0; seq < length; seq++)
        {
            for (int hidden = 0; hidden < hiddenSize; hidden++)
            {
                slice[0, seq, hidden] = embeddings[0, startPos + seq, hidden];
            }
        }
        
        return slice;
    }

    /// <summary>
    /// Combines embeddings in LLaVA style: [pre_text] + [image_features] + [post_text]
    /// FIXED: This now properly inserts image features at the image token position
    /// matching the working Rust implementation logic
    /// </summary>
    private DenseTensor<float> CombineEmbeddingsLlavaStyle(Tensor<float> preTextEmbeddings, Tensor<float> imageFeatures, Tensor<float> postTextEmbeddings)
    {
        int hiddenSize = (int)imageFeatures.Dimensions[2]; // Should be 896 for FastVLM
        
        // Calculate sequence lengths
        int preTextLen = preTextEmbeddings != null ? (int)preTextEmbeddings.Dimensions[1] : 0;
        int imageLen = (int)imageFeatures.Dimensions[1]; // 49 tokens
        int postTextLen = postTextEmbeddings != null ? (int)postTextEmbeddings.Dimensions[1] : 0;
        
        // CRITICAL FIX: The total length should be pre + image + post WITHOUT the image token
        // because image features REPLACE the image token, not add to it
        int totalLength = preTextLen + imageLen + postTextLen;
        
        Debug.WriteLine($"Fusion breakdown: pre={preTextLen}, image={imageLen}, post={postTextLen}, total={totalLength}");
        
        var combined = new DenseTensor<float>(new[] { 1, totalLength, hiddenSize });
        int currentPos = 0;
        
        // 1. Copy pre-text embeddings (everything before image token)
        if (preTextEmbeddings != null && preTextLen > 0)
        {
            for (int seq = 0; seq < preTextLen; seq++)
            {
                for (int hidden = 0; hidden < hiddenSize; hidden++)
                {
                    combined[0, currentPos + seq, hidden] = preTextEmbeddings[0, seq, hidden];
                }
            }
            currentPos += preTextLen;
            Debug.WriteLine($"Copied pre-text: {preTextLen} tokens at position 0-{preTextLen-1}");
        }
        
        // 2. Insert image features (replacing the image token)
        for (int seq = 0; seq < imageLen; seq++)
        {
            for (int hidden = 0; hidden < hiddenSize; hidden++)
            {
                combined[0, currentPos + seq, hidden] = imageFeatures[0, seq, hidden];
            }
        }
        currentPos += imageLen;
        Debug.WriteLine($"Inserted image features: {imageLen} tokens at position {currentPos-imageLen}-{currentPos-1}");
        
        // 3. Copy post-text embeddings (everything after image token) 
        if (postTextEmbeddings != null && postTextLen > 0)
        {
            for (int seq = 0; seq < postTextLen; seq++)
            {
                for (int hidden = 0; hidden < hiddenSize; hidden++)
                {
                    combined[0, currentPos + seq, hidden] = postTextEmbeddings[0, seq, hidden];
                }
            }
            Debug.WriteLine($"Copied post-text: {postTextLen} tokens at position {currentPos}-{currentPos+postTextLen-1}");
        }
        
        // Verify image features were preserved
        float imageSum = 0f;
        for (int i = 0; i < Math.Min(5, imageLen); i++)
        {
            imageSum += Math.Abs(combined[0, preTextLen + i, 0]);
        }
        Debug.WriteLine($"Image feature verification - first 5 tokens sum: {imageSum:F2}");
        
        Debug.WriteLine($"Final combined embeddings: {totalLength} total tokens, {hiddenSize} hidden size");
        return combined;
    }

    public void Dispose()
    {
        _enc?.Dispose();
        _emb?.Dispose();
        _dec?.Dispose();
    }
}
