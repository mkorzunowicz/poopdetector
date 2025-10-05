// ------------------------------------------------------------
//  EdgeSamImageProcessor.cs
// ------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;
using SkiaSharp;
using PointF = Microsoft.Maui.Graphics.PointF;
using Size = Microsoft.Maui.Graphics.Size;

namespace PoopDetector.AI.Vision.EdgeSam
{
    /// <summary>
    /// EdgeSAM Image Processor
    /// 
    /// Similar to RepViT-SAM:
    /// • Fixed 1024×1024 input size
    /// • Uses ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    /// • Produces float32 tensor [1,3,1024,1024] in NCHW format (channel-first)
    /// • Simpler decoder inputs (only 3 tensors)
    /// </summary>
    public sealed class EdgeSamImageProcessor
        : SkiaSharpImageProcessor<SKBitmap, float>
    {
        private const int TargetSize = 1024;  // EdgeSAM uses fixed 1024×1024

        // Parameterless constructor required by DoubleVisionBase
        public EdgeSamImageProcessor() { }

        // Fixed size requirement
        public override Size RequiredSize => new Size(TargetSize, TargetSize);

        // --------------------------------------------------------------------
        // 0.  No-op preprocessing – just pass original bitmap through
        // --------------------------------------------------------------------
        protected override SKBitmap OnPreprocessSourceImage(SKBitmap src) => src;

        // --------------------------------------------------------------------
        // 1.  Convert SKBitmap → [1,3,1024,1024] float32 tensor (NCHW format)
        //     EdgeSAM expects fixed 1024×1024 input in NCHW format
        //     with pixel values normalized: (pixel / 255.0 - mean) / std
        //     ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        // --------------------------------------------------------------------
        protected override Tensor<float> OnGetTensorForImage(SKBitmap bmpIn)
        {
            // A) resize to 1024×1024 (stretching if needed)
            var infoResized = new SKImageInfo(TargetSize, TargetSize,
                                              SKColorType.Rgba8888,
                                              SKAlphaType.Unpremul);

            using SKBitmap resized = bmpIn.Resize(infoResized, SKFilterQuality.Medium)
                                      ?? throw new InvalidOperationException("Resize failed");

            // B) ImageNet normalization constants
            float[] means = new[] { 0.485f, 0.456f, 0.406f };  // RGB
            float[] stds = new[] { 0.229f, 0.224f, 0.225f };   // RGB

            // C) Create NCHW tensor [1, 3, 1024, 1024]
            float[] data = new float[1 * 3 * TargetSize * TargetSize];
            SKColor[] pix = resized.Pixels;  // RGBA order, Unpremul

            int channelSize = TargetSize * TargetSize;
            
            // Fill in channel-first (NCHW) format with normalization
            for (int y = 0; y < TargetSize; y++)
            {
                for (int x = 0; x < TargetSize; x++)
                {
                    int pixelIdx = y * TargetSize + x;
                    var c = pix[pixelIdx];
                    
                    // Normalize: (pixel / 255.0 - mean) / std
                    data[0 * channelSize + pixelIdx] = ((c.Red / 255.0f) - means[0]) / stds[0];      // R channel
                    data[1 * channelSize + pixelIdx] = ((c.Green / 255.0f) - means[1]) / stds[1];    // G channel
                    data[2 * channelSize + pixelIdx] = ((c.Blue / 255.0f) - means[2]) / stds[2];     // B channel
                }
            }

            return new DenseTensor<float>(data, new[] { 1, 3, TargetSize, TargetSize });
        }

        // --------------------------------------------------------------------
        // 2.  Build decoder inputs from points
        // --------------------------------------------------------------------
        public IReadOnlyCollection<NamedOnnxValue> BuildDecoderInputs(
            float[] embedding,
            IReadOnlyList<PointF> points,
            Size origSize)
        {
            // Embedding dimensions for EdgeSAM
            var embTensor = new DenseTensor<float>(embedding, new[] { 1, 256, 64, 64 });

            int numPoints = points.Count;
            var coords = new DenseTensor<float>(new[] { 1, numPoints, 2 });
            var labels = new DenseTensor<float>(new[] { 1, numPoints });

            // Map points from original image space to 1024x1024 encoder space
            for (int i = 0; i < numPoints; i++)
            {
                coords[0, i, 0] = points[i].X;  // X already in encoder space
                coords[0, i, 1] = points[i].Y;  // Y already in encoder space
                labels[0, i] = 1f;              // foreground point
            }

            return CommonDecoderInputs(embTensor, coords, labels, origSize);
        }

        // --------------------------------------------------------------------
        // 3.  Build decoder inputs from bounding box
        // --------------------------------------------------------------------
        public IReadOnlyCollection<NamedOnnxValue> BuildDecoderInputs(
            float[] embedding,
            RectangleF box,
            Size origSize)
            => BuildDecoderInputs(
                 embedding,
                 new[] {
                     new PointF(box.Left,  box.Top),
                     new PointF(box.Right, box.Bottom)
                 },
                 origSize);

        // --------------------------------------------------------------------
        // 4.  Convert decoder output → Gray8 SKBitmap, resize to original dimensions
        // --------------------------------------------------------------------
        public SKBitmap PostprocessMask(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> res,
            Size originalSize,
            float threshold = 0f)
        {
            var maskTensor = res.First(o => o.Name.Contains("masks")).AsTensor<float>();
            int h = maskTensor.Dimensions[2];
            int w = maskTensor.Dimensions[3];

            float[] src = maskTensor.ToArray();
            byte[] dst = new byte[w * h];

            for (int i = 0; i < dst.Length; i++)
            {
                // logits -> probability
                float p = 1f / (1f + MathF.Exp(-src[i]));
                dst[i] = p > 0.5f ? (byte)255 : (byte)0;
            }

            var bmp = new SKBitmap(w, h, SKColorType.Gray8, SKAlphaType.Opaque);

            // pin -> copy -> unpin
            var handle = System.Runtime.InteropServices.GCHandle
                         .Alloc(dst, System.Runtime.InteropServices.GCHandleType.Pinned);
            try
            {
                bmp.InstallPixels(bmp.Info, handle.AddrOfPinnedObject(), w);
            }
            finally
            {
                handle.Free();
            }

            // Resize back to original dimensions to fix stretching
            if (originalSize.Width != w || originalSize.Height != h)
            {
                int targetWidth = (int)originalSize.Width;
                int targetHeight = (int)originalSize.Height;
                
                var resized = new SKBitmap(targetWidth, targetHeight, 
                                          SKColorType.Gray8, SKAlphaType.Opaque);
                using (var canvas = new SKCanvas(resized))
                {
                    canvas.Clear(SKColors.Black);
                    canvas.DrawBitmap(bmp, 
                        new SKRect(0, 0, (float)originalSize.Width, (float)originalSize.Height),
                        new SKPaint { FilterQuality = SKFilterQuality.Medium });
                }
                bmp.Dispose();
                return resized;
            }

            return bmp;
        }

        // --------------------------------------------------------------------
        // helpers
        // --------------------------------------------------------------------
        private IReadOnlyCollection<NamedOnnxValue> CommonDecoderInputs(
            DenseTensor<float> embTensor,
            DenseTensor<float> coordsTensor,
            DenseTensor<float> labelsTensor,
            Size origSize)
        {
            // Based on the screenshot:
            // Inputs: image_embeddings, point_coords, point_labels
            // Outputs: masks, scores
            
            return new[]
            {
                NamedOnnxValue.CreateFromTensor("image_embeddings", embTensor),
                NamedOnnxValue.CreateFromTensor("point_coords",     coordsTensor),
                NamedOnnxValue.CreateFromTensor("point_labels",     labelsTensor)
            };
        }
    }
}
