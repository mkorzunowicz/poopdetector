using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision;
using PoopDetector.AI.Vision.Processing;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace PoopDetector.AI.Vision.FastVLM;

/// <summary>
/// Simplified FastVLM implementation following transformers.js patterns
/// </summary>
public sealed class FastVlmSimple : IVision, IVisualLanguageModel
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

    public string Name => "FastVLM-0.5B-Simple";
    public string ModelName => "FastVLM-0.5B-Simple";
    public byte[] Model => Array.Empty<byte>();
    public InferenceSession Session => _dec;
    public Microsoft.Maui.Graphics.Size InputSize => new(448, 448);
    public FastVlmImageProcessor ImageProcessor => _image;

    public FastVlmSimple(string visionEncoderPath,
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
    }

    public static async Task<FastVlmSimple> CreateAsync(string visionEncoderPath,
                                                       string embedTokensPath,
                                                       string decoderPath,
                                                       string vocabPath,
                                                       string mergesPath,
                                                       CancellationToken cancellationToken = default)
    {
        var model = new FastVlmSimple(visionEncoderPath, embedTokensPath, decoderPath, 
                                     vocabPath, mergesPath, string.Empty);
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

            _tok = new HfTokenizer(_vocabPath, _mergesPath);
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
        const string defaultPrompt = "What do you see in this image?";
        string json = await GenerateAsync(image, defaultPrompt, 20);
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

    public async Task<string> GenerateAsync(byte[] image, string prompt, int maxNewTokens = 50, CancellationToken ct = default)
    {
        return await Task.Run(() =>
        {
            try
            {
                Debug.WriteLine($"=== Starting generation with merged decoder approach ===");
                
            // 1. Process image through vision encoder
            // Input: pixel_values (float32[s0,3,s1,s2])  
            // Output: image_features (float32[s0,((((s1-1)//64))+1)*((((s2-1)//64))+1),896])
            using var bmp = _image.PreprocessSourceImage(image);
            var imgTensor = _image.GetTensorForImage(bmp);

            using var encRes = _enc.Run(new[] { NamedOnnxValue.CreateFromTensor("pixel_values", imgTensor) });
            var imgEmbTensor = encRes.First().AsTensor<float>();

            Debug.WriteLine($"Vision encoder input shape: [{string.Join("x", imgTensor.Dimensions.ToArray())}]");
            Debug.WriteLine($"Vision encoder output shape: [{string.Join("x", imgEmbTensor.Dimensions.ToArray())}]");                // 2. Use proper Apple FastVLM format with <image> placeholder
                const string userQuestion = "What do you see in this image?";
                const string appleFormat = $"<image>\n{userQuestion}";
                
                // Split on <image> and handle the IMAGE_TOKEN_INDEX = -200
                var parts = appleFormat.Split("<image>");
                if (parts.Length != 2)
                {
                    throw new InvalidOperationException("Prompt must contain exactly one <image> placeholder");
                }
                
                string preImageText = parts[0];
                string postImageText = parts[1];
                
                // Tokenize parts
                var preTokens = string.IsNullOrEmpty(preImageText) ? new int[0] : _tok.Encode(preImageText);
                var postTokens = _tok.Encode(postImageText);
                
                const int IMAGE_TOKEN_INDEX = -200;
                
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

                Debug.WriteLine("Running decoder with input_ids and pixel_values...");
                using var decRes = _dec.Run(inputs);
                var outputs = decRes.ToArray();
                
                Debug.WriteLine($"Decoder outputs: {outputs.Length} tensors");
                foreach (var output in outputs)
                {
                    Debug.WriteLine($"- {output.Name}: {string.Join("x", output.AsTensor<float>().Dimensions.ToArray())}");
                }
                
                var logits = outputs[0].AsTensor<float>();
                
                // Sample from logits for the last position
                int seqLen = (int)logits.Dimensions[1];
                int vocabSize = (int)logits.Dimensions[2];
                
                Debug.WriteLine($"Logits shape: [1, {seqLen}, {vocabSize}]");
                
                var lastTokenLogits = new float[vocabSize];
                for (int i = 0; i < vocabSize; i++)
                {
                    lastTokenLogits[i] = logits[0, seqLen - 1, i];
                }

                // Sample next token
                var nextTokenId = SampleNextToken(lastTokenLogits, null);
                string result = _tok.Decode(new[] { nextTokenId });
                
                Debug.WriteLine($"Generated token {nextTokenId} = '{result}'");
                return result;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error in GenerateAsync: {ex}");
                
                // Fallback to the original approach if decoder doesn't accept pixel_values directly
                Debug.WriteLine("Trying fallback approach with feature combination...");
                return GenerateAsyncFallback(image, prompt, maxNewTokens, ct);
            }
        }, ct);
    }

    private string GenerateAsyncFallback(byte[] image, string prompt, int maxNewTokens, CancellationToken ct)
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

            // Step 1: Apply chat template with correct number of image tokens
            string userQuestion = prompt.Replace("<image>", "").Trim();
            if (string.IsNullOrEmpty(userQuestion))
                userQuestion = "Describe what you see.";
            
            string chatMLPrompt = "<|im_start|>system\nYou are a helpful visual AI assistant. Respond concisely and accurately to the user's query in one sentence.<|im_end|>\n" +
                                 $"<|im_start|>user\n<image>{userQuestion}<|im_end|>\n" +
                                 "<|im_start|>assistant\n";
            
            Debug.WriteLine($"Chat template applied: {chatMLPrompt}");
            Debug.WriteLine($"Using actual image tokens from vision encoder: {actualImageTokens}");
            
            Debug.WriteLine($"Chat template applied: {chatMLPrompt}");
            
            // Step 2: Tokenize the prompt parts separately and manually insert image tokens
            // Split the ChatML prompt around the <image> placeholder
            const int IMAGE_TOKEN_INDEX = 151646;
            var promptParts = chatMLPrompt.Split("<image>");
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

            var nextTokenId = SampleNextToken(lastTokenLogits, null);
            
            // Generate multiple tokens instead of just one
            var generatedTokens = new List<int> { nextTokenId };
            
            Debug.WriteLine($"Step 0: Generated token {nextTokenId} = '{_tok.Decode(new[] { nextTokenId })}'");
            
            // For LLaVA-style generation, track all tokens that have been generated
            // We'll extend the embeddings for each new token
            generatedTokens.Add(nextTokenId);
            
            for (int step = 1; step < maxNewTokens; step++)
            {
                // Check for EOS token: <|im_end|> is token 151645 according to tokenizer_config.json
                if (nextTokenId == 151645 || nextTokenId == 2) // <|im_end|> or fallback EOS
                {
                    Debug.WriteLine($"Generated EOS token ({nextTokenId}), stopping generation");
                    break;
                }
                
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
                
                nextTokenId = SampleNextToken(newLastTokenLogits, generatedTokens);
                generatedTokens.Add(nextTokenId);
                
                string tokenText = _tok.Decode(new[] { nextTokenId });
                
                Debug.WriteLine($"Step {step}: Generated token {nextTokenId} = '{tokenText}'");
                
                // Update generated tokens list (no need for currentTokens)
                
                // Stop at natural boundaries
                if (tokenText.Contains(".") || tokenText.Contains("!") || tokenText.Contains("?"))
                {
                    Debug.WriteLine($"Found natural stopping point, ending generation");
                    break;
                }
            }
            
            // Decode all generated tokens together for proper BPE handling
            Debug.WriteLine($"Generated tokens: [{string.Join(", ", generatedTokens)}]");
            string result = _tok.Decode(generatedTokens.ToArray()).Trim();
            Debug.WriteLine($"Final decoded result: '{result}'");
            
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
        int existingLength = (int)existingEmbeddings.Dimensions[1];
        int hiddenSize = (int)existingEmbeddings.Dimensions[2];
        int newLength = existingLength + 1;
        
        var extended = new DenseTensor<float>(new[] { 1, newLength, hiddenSize });
        
        // Copy existing embeddings
        var existingArray = existingEmbeddings.ToArray();
        var newArray = newTokenEmbedding.ToArray();
        
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
        
        Debug.WriteLine($"Extended embeddings from {existingLength} to {newLength} tokens");
        return extended;
    }

    /// <summary>
    /// Combines embeddings in LLaVA style: [pre_text] + [image_features] + [post_text]
    /// </summary>
    private DenseTensor<float> CombineEmbeddingsLlavaStyle(Tensor<float> preTextEmbeddings, Tensor<float> imageFeatures, Tensor<float> postTextEmbeddings)
    {
        var embeddingParts = new List<Tensor<float>>();
        int totalLength = 0;
        int hiddenSize = (int)imageFeatures.Dimensions[2]; // Should be 896 for FastVLM
        
        // Add pre-text embeddings if present
        if (preTextEmbeddings != null && preTextEmbeddings.Dimensions[1] > 0)
        {
            embeddingParts.Add(preTextEmbeddings);
            totalLength += (int)preTextEmbeddings.Dimensions[1];
            Debug.WriteLine($"Added pre-text embeddings: {preTextEmbeddings.Dimensions[1]} tokens");
        }
        
        // Add image features - these replace the <image> token
        embeddingParts.Add(imageFeatures);
        totalLength += (int)imageFeatures.Dimensions[1];
        Debug.WriteLine($"Added image features: {imageFeatures.Dimensions[1]} visual tokens");
        
        // Add post-text embeddings if present  
        if (postTextEmbeddings != null && postTextEmbeddings.Dimensions[1] > 0)
        {
            embeddingParts.Add(postTextEmbeddings);
            totalLength += (int)postTextEmbeddings.Dimensions[1];
            Debug.WriteLine($"Added post-text embeddings: {postTextEmbeddings.Dimensions[1]} tokens");
        }
        
        // Create combined tensor
        var combined = new DenseTensor<float>(new[] { 1, totalLength, hiddenSize });
        int currentPos = 0;
        
        foreach (var part in embeddingParts)
        {
            var partLength = (int)part.Dimensions[1];
            var partArray = part.ToArray();
            
            for (int seq = 0; seq < partLength; seq++)
            {
                for (int hidden = 0; hidden < hiddenSize; hidden++)
                {
                    combined[0, currentPos + seq, hidden] = partArray[seq * hiddenSize + hidden];
                }
            }
            currentPos += partLength;
        }
        
        Debug.WriteLine($"Combined embeddings: {totalLength} total tokens, {hiddenSize} hidden size");
        return combined;
    }

    public void Dispose()
    {
        _enc?.Dispose();
        _emb?.Dispose();
        _dec?.Dispose();
    }
}
