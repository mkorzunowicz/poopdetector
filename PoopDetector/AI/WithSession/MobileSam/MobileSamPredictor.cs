//using System;
//using System.Collections.Generic;
//using System.Diagnostics;
//using System.Linq;
//using Microsoft.ML.OnnxRuntime;
//using Microsoft.ML.OnnxRuntime.Tensors;
//using SkiaSharp;

//namespace MobileSAM
//{
//    public class SamPredictor : IDisposable
//    {
//        private InferenceSession _encoderSession;
//        private InferenceSession _decoderSession;
//        private ResizeLongestSide _transform;

//        // Store the final [1, C, H_emb, W_emb] image embeddings, typically shape [1,256,64,64].
//        private float[] _imageEmbeddingData;
//        private int _embeddingH, _embeddingW, _embeddingC;

//        // We keep track of the original image size so we can transform prompts.
//        private (int h, int w) _originalImageSize;
//        private (int h, int w) _modelInputImageSize;

//        // If you only have a single "decoder" ONNX, remove the encoder logic below.
//        public SamPredictor(string encoderOnnxPath, string decoderOnnxPath, int encoderInputLongSide = 1024)
//        {
//            _encoderSession = new InferenceSession(encoderOnnxPath);
//            _decoderSession = new InferenceSession(decoderOnnxPath);

//            // Sam uses 1024 by default for the longest side. 
//            _transform = new ResizeLongestSide(encoderInputLongSide);
//        }

//        /// <summary>
//        /// If you only have the decoder, you might skip this constructor and supply the embedding yourself.
//        /// </summary>
//        public SamPredictor(InferenceSession decoderSession, int encoderInputLongSide = 1024)
//        {
//            _decoderSession = decoderSession;
//            _transform = new ResizeLongestSide(encoderInputLongSide);
//        }

//        /// <summary>
//        /// SetImage: loads an RGB image (SkiaSharp-based), resizes it to the model input,
//        /// and runs the "image encoder" ONNX to produce image_embeddings.
//        /// Then we store them for subsequent calls to Predict(...).
//        /// </summary>
//        public void SetImage(SKBitmap originalImage)
//        {
//            _originalImageSize = (originalImage.Height, originalImage.Width);

//            // 1) Resize to e.g. 1024 on the long side
//            SKBitmap resized = _transform.ApplyImage(originalImage);
//            _modelInputImageSize = (resized.Height, resized.Width); // e.g. (someH, someW)

//            // 2) Convert to CHW float array
//            float[] inputData = PrepareEncoderInput(resized);

//            // 3) Run the encoder session => get [1,C,H_emb,W_emb] image embeddings
//            if (_encoderSession == null)
//            {
//                throw new InvalidOperationException("No encoder session. Provide an encoder ONNX or do your embedding externally.");
//            }

//            // Build input
//            var inputMeta = _encoderSession.InputMetadata;
//            string inputName = inputMeta.Keys.First();
//            // Typically "x" or "images" or something; check your model.

//            // For example, shape = [1,3,1024,1024]
//            var tensorIn = new DenseTensor<float>(inputData, new[] { 1, 3, _modelInputImageSize.h, _modelInputImageSize.w });
//            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensorIn) };

//            using var results = _encoderSession.Run(inputs);
//            // The output might be named "output", "image_embeddings", or something else
//            var firstOutput = results.First();
//            float[] embeddingArray = firstOutput.AsEnumerable<float>().ToArray();

//            // Suppose shape is [1, 256, 64, 64]:
//            // We can store them for later usage
//            // (You might parse shape from model metadata if needed.)
//            _embeddingC = 256;
//            _embeddingH = 64;
//            _embeddingW = 64;
//            // Adjust if your model uses different dims

//            _imageEmbeddingData = embeddingArray;
//        }

//        /// <summary>
//        /// Returns the [1,C,H_emb,W_emb] embedding we computed from SetImage.
//        /// If you only have a PyTorch embedding, you'd store that directly here as float[].
//        /// </summary>
//        public float[] GetImageEmbedding()
//        {
//            if (_imageEmbeddingData == null || _imageEmbeddingData.Length == 0)
//                throw new InvalidOperationException("No image embedding found. Call SetImage(...) or provide embedding data.");
//            return _imageEmbeddingData;
//        }

//        /// <summary>
//        /// The main call that replicates "masks, _, low_res_logits = ort_session.run(...)" in Python.
//        /// Pass in point coords, labels, optional mask_input, etc.
//        /// Returns a float[] mask of shape [B, 1, H, W] or [B,3,H,W] depending on multi-mask.
//        /// (In practice, you'll parse shapes carefully.)
//        /// 
//        /// - pointCoords/pointLabels shape = Nx2, Nx1
//        /// - boxInput shape = Nx4
//        /// - maskInput shape = Nx1x256x256
//        /// - hasMaskInput shape = Nx1
//        /// </summary>
//        public (float[] masks, float[] lowResLogits) Predict(
//            (float x, float y)[] pointCoords,
//            int[] pointLabels,
//            float[] maskInput = null,
//            bool multiMaskOutput = false,
//            (float x1, float y1, float x2, float y2)? boxInput = null)
//        {
//            if (_decoderSession == null)
//                throw new InvalidOperationException("No decoder session available.");

//            // 1) Transform the prompts from original image space to the resized image space
//            //    using the same ResizeLongestSide transform.
//            //    Because SamPredictor does "apply_coords" or "apply_boxes"
//            //    so that the prompts match the encoder size.
//            var transformedPoints = _transform.ApplyCoords(pointCoords.Select(p => ((float)p.x, (float)p.y)).ToArray(), _originalImageSize);

//            // 2) Build ONNX inputs
//            //    In the Python snippet, we do:
//            //      onnx_coord   => shape [1, N+1, 2] (the +1 is for the dummy prompt)
//            //      onnx_label   => shape [1, N+1]
//            //      onnx_mask_input => shape [1,1,256,256]
//            //      has_mask_input  => shape [1]
//            //      orig_im_size    => shape [2]
//            //      image_embeddings => shape [1,C,H,W]

//            //   Because MobileSAM adds a dummy coordinate and label (-1). We'll replicate that.

//            int N = transformedPoints.Length;
//            int totalPoints = N + 1; // extra “dummy” point

//            float[] coordsArray = new float[totalPoints * 2];
//            float[] labelArray = new float[totalPoints];

//            // Copy real points
//            for (int i = 0; i < N; i++)
//            {
//                coordsArray[i * 2 + 0] = transformedPoints[i].x;
//                coordsArray[i * 2 + 1] = transformedPoints[i].y;
//                labelArray[i] = pointLabels[i];
//            }
//            // Add dummy point with label -1
//            coordsArray[(totalPoints - 1) * 2 + 0] = 0f;
//            coordsArray[(totalPoints - 1) * 2 + 1] = 0f;
//            labelArray[totalPoints - 1] = -1f;

//            // Now shape = [1, totalPoints, 2]
//            var pointCoordsTensor = new DenseTensor<float>(coordsArray, new[] { 1, totalPoints, 2 });
//            var pointLabelsTensor = new DenseTensor<float>(labelArray, new[] { 1, totalPoints });

//            // Box input if given
//            float[] boxArr = null;
//            if (boxInput.HasValue)
//            {
//                // transform box
//                var boxVal = boxInput.Value;
//                var boxTransformed = _transform.ApplyBoxes(new[] { (boxVal.x1, boxVal.y1, boxVal.x2, boxVal.y2) }, _originalImageSize);
//                boxArr = new float[4] { boxTransformed[0].x1, boxTransformed[0].y1, boxTransformed[0].x2, boxTransformed[0].y2 };
//            }

//            // maskInput => shape = [1,1,256,256], or 0s if none
//            // hasMask => shape = [1]
//            float[] maskInputArray = null;
//            float[] hasMaskArray = new float[1] { 0f };

//            if (maskInput != null && maskInput.Length == (1 * 256 * 256))
//            {
//                maskInputArray = maskInput;
//                hasMaskArray[0] = 1f;
//            }
//            else
//            {
//                // Provide a blank [1,1,256,256]
//                maskInputArray = new float[1 * 256 * 256];
//            }

//            var maskInputTensor = new DenseTensor<float>(maskInputArray, new[] { 1, 1, 256, 256 });
//            var hasMaskTensor = new DenseTensor<float>(hasMaskArray, new[] { 1 });

//            // 3) multiMaskOutput => the ONNX might rely on a boolean or integer flag
//            //    In some ONNX exports, you pass "orig_im_size", "mask_input", "point_coords", 
//            //    "point_labels", "image_embeddings", "has_mask_input", 
//            //    and possibly "init_prompt", etc. 
//            //    The exact names and usage depend on your exported model.

//            // We'll do a typical name set (you must confirm them by checking your ONNX):
//            var ortInputs = new List<NamedOnnxValue>();

//            // image_embeddings => shape [1,C,64,64]
//            var embeddingsTensor = new DenseTensor<float>(
//                _imageEmbeddingData, new[] { 1, _embeddingC, _embeddingH, _embeddingW }
//            );

//            ortInputs.Add(NamedOnnxValue.CreateFromTensor("image_embeddings", embeddingsTensor));
//            ortInputs.Add(NamedOnnxValue.CreateFromTensor("point_coords", pointCoordsTensor));
//            ortInputs.Add(NamedOnnxValue.CreateFromTensor("point_labels", pointLabelsTensor));
//            ortInputs.Add(NamedOnnxValue.CreateFromTensor("mask_input", maskInputTensor));
//            ortInputs.Add(NamedOnnxValue.CreateFromTensor("has_mask_input", hasMaskTensor));

//            // Original image size => shape [2]
//            float[] origImSizeArray = new float[2] { _originalImageSize.h, _originalImageSize.w };
//            var origImSizeTensor = new DenseTensor<float>(origImSizeArray, new[] { 2 });
//            ortInputs.Add(NamedOnnxValue.CreateFromTensor("orig_im_size", origImSizeTensor));

//            // If your decoder expects “orig_im_size” as shape [1,2], then adapt accordingly:
//            // var origImSizeTensor = new DenseTensor<float>(origImSizeArray, new[] {1,2});

//            // If it expects a "multimask_output" or something, you'd pass that as well, 
//            // but frequently you’d export a version that always returns 3 masks by default.

//            if (boxArr != null)
//            {
//                var boxTensor = new DenseTensor<float>(boxArr, new[] { 1, 4 });
//                ortInputs.Add(NamedOnnxValue.CreateFromTensor("boxes", boxTensor));
//            }

//            // 4) Run inference
//            using var results = _decoderSession.Run(ortInputs);

//            // Usually the model returns [masks, scores, low_res_logits], or something similar.
//            // e.g. "masks" => shape [1, 1, H, W], or [1,3,H,W].
//            // Let's assume your model returns 3 outputs in a fixed order, and the first is "masks".
//            // Or we check by name:
//            var outputsArr = results.ToArray();

//            // Let's attempt to find by name "masks" or just take outputsArr[0].
//            var masksOutput = outputsArr.First(o => o.Name.Contains("masks"));
//            var lowResOutput = outputsArr.FirstOrDefault(o => o.Name.Contains("low_res_logits"));
//            // Possibly there's a "scores" or "iou_predictions" as well.

//            float[] maskData = masksOutput.AsEnumerable<float>().ToArray();
//            float[] lowResData = lowResOutput != null
//                ? lowResOutput.AsEnumerable<float>().ToArray()
//                : new float[0];

//            // The shape might be [1, 3, H, W] if multiMask = true,
//            // or [1,1,H,W] if single mask. 
//            // You can parse shapes from the Onnx output metadata if needed.

//            return (maskData, lowResData);
//        }

//        /// <summary>
//        /// Prepare the encoder input from a SkiaSharp bitmap: 
//        ///   - Convert RGBA to float[3, H, W] in (R,G,B) order
//        ///   - Possibly subtract mean and divide by std if your model requires
//        /// </summary>
//        private float[] PrepareEncoderInput(SKBitmap resized)
//        {
//            int w = resized.Width;
//            int h = resized.Height;

//            // Typically SAM uses mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375]
//            // Check your MobileSAM export for specifics.
//            float[] MEAN = { 123.675f, 116.28f, 103.53f };
//            float[] STD = { 58.395f, 57.12f, 57.375f };

//            float[] tensorData = new float[3 * h * w];
//            var pixels = resized.Pixels; // RGBA

//            int hw = h * w;
//            for (int i = 0; i < hw; i++)
//            {
//                uint rgba = (uint)pixels[i];
//                byte r = (byte)((rgba >> 0) & 0xFF);
//                byte g = (byte)((rgba >> 8) & 0xFF);
//                byte b = (byte)((rgba >> 16) & 0xFF);
//                // alpha is (byte)((rgba >> 24) & 0xFF), if needed

//                // channel-first
//                float fr = (r - MEAN[0]) / STD[0];
//                float fg = (g - MEAN[1]) / STD[1];
//                float fb = (b - MEAN[2]) / STD[2];

//                tensorData[0 * hw + i] = fr;
//                tensorData[1 * hw + i] = fg;
//                tensorData[2 * hw + i] = fb;
//            }

//            return tensorData;
//        }

//        public void Dispose()
//        {
//            _encoderSession?.Dispose();
//            _decoderSession?.Dispose();
//        }
//    }
//}
