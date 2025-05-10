// ------------------------------------------------------------
//  MobileSamImageProcessor.cs
// ------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Net.NetworkInformation;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;
using SkiaSharp;
using PointF = Microsoft.Maui.Graphics.PointF;
using Size = Microsoft.Maui.Graphics.Size;  

namespace PoopDetector.AI.Vision.MobileSam
{
    /// <summary>
    /// • NO resizing / padding – the encoder graph does that internally.  
    /// • Produces a     float32 tensor  [1,3,H,W]  with raw RGB 0-255.  
    /// • Calculates the virtual (w_res,h_res) that the encoder will see
    ///   (long side = 1024, other side scaled) so prompt coordinates can
    ///   be mapped properly.
    /// </summary>
    public sealed class MobileSamImageProcessor
        : SkiaSharpImageProcessor<SKBitmap, float>
    {
        private const int TargetLongSide = 1024;

        // no external Resize requirement
        public override Size RequiredSize => Size.Zero;

        // --------------------------------------------------------------------
        //   help the caller map prompts by telling what encoder size will be
        // --------------------------------------------------------------------
        public Size GetEncoderSize(Size orig)
        {
            if (orig.Width >= orig.Height)
            {
                int w = TargetLongSide;
                int h = (int)MathF.Round((float)orig.Height / (float)orig.Width * TargetLongSide);
                return new Size(w, h);
            }
            else
            {
                int h = TargetLongSide;
                int w = (int)MathF.Round((float)orig.Width / (float)orig.Height * TargetLongSide);
                return new Size(w, h);
            }
        }
        // quick helper for MobileSam.cs
        public Size GetEncoderSize(SKBitmap bmp) => GetEncoderSize(
            new Size(bmp.Width, bmp.Height));

        // expose for MobileSam.cs
        public Size GetEncoderSize(int w, int h) => GetEncoderSize(
            new Size(w, h));

        // --------------------------------------------------------------------
        // 0.  No-op preprocessing – just pass original bitmap through
        // --------------------------------------------------------------------
        protected override SKBitmap OnPreprocessSourceImage(SKBitmap src) => src;

        // --------------------------------------------------------------------
        // 1.  Convert SKBitmap →  [1,3,H,W] float32 tensor (RGB order)
        // --------------------------------------------------------------------
        protected override Tensor<float> OnGetTensorForImage(SKBitmap bmpIn)
        {
            // TOOD: check if the Target and origSize makes sense here
            // It works, so i kinda down't want to change it anymore
            const int Target = 1024;                       // SAM’s expected size

            // A) find resized dimensions that keep aspect ratio
            int newW, newH;
            if (bmpIn.Width >= bmpIn.Height)
            {
                newW = Target;
                newH = (int)MathF.Round((float)bmpIn.Height / bmpIn.Width * Target);
            }
            else
            {
                newH = Target;
                newW = (int)MathF.Round((float)bmpIn.Width / bmpIn.Height * Target);
            }

            // B) resize original to (newW,newH) in RGBA8888 - Unpremul
            var infoResized = new SKImageInfo(newW, newH,
                                              SKColorType.Rgba8888,
                                              SKAlphaType.Unpremul);

            using SKBitmap resized = bmpIn.Resize(infoResized, SKFilterQuality.Medium)
                                      ?? throw new InvalidOperationException("Resize failed");

            // C) put it into the top-left corner of a 1024×1024 black canvas
            var infoFull = new SKImageInfo(Target, Target,
                                           SKColorType.Rgba8888,
                                           SKAlphaType.Unpremul);

            using SKBitmap full = new SKBitmap(infoFull);
            full.Erase(SKColors.Black);                    // zero-pad
            using (var canvas = new SKCanvas(full))
            {
                canvas.DrawBitmap(resized, 0, 0);
            }

            // D) channel-last NHWC tensor
            float[] data = new float[Target * Target * 3];
            int idx = 0;
            SKColor[] pix = full.Pixels;                   // RGBA order, Unpremul
            for (int i = 0; i < pix.Length; i++)
            {
                var c = pix[i];
                data[idx++] = c.Red;
                data[idx++] = c.Green;
                data[idx++] = c.Blue;
            }

            // (optional) store the actually-resized size for coord mapping
            //ResizedWidth = newW;
            //ResizedHeight = newH;

            return new DenseTensor<float>(data, new[] { Target, Target, 3 });
        }
        // --------------------------------------------------------------------
        // 2.  Build decoder inputs  (points overload)
        // --------------------------------------------------------------------
        public IReadOnlyCollection<NamedOnnxValue> BuildDecoderInputs(
            float[] embedding,
            IReadOnlyList<PointF> points,
            Size encSize,
            Size origSize)
        {
            var embTensor = new DenseTensor<float>(
                embedding, new[] { 1, 256, 64, 64 });

            // map points from original space → encoder (w_res,h_res),
            // then into 1024-padded space
            //float sx = (float)encSize.Width / (float)origSize.Width;
            //float sy = (float)encSize.Height / (float)origSize.Height;
            float sx = 1;
            float sy = 1;
            int N = points.Count;
            var coords = new float[(N + 1) * 2];
            var labels = new float[N + 1];

            for (int i = 0; i < N; i++)
            {
                coords[2 * i + 0] = points[i].X * sx;
                coords[2 * i + 1] = points[i].Y * sy;
                labels[i] = 1;           // foreground
            }
            coords[2 * N] = coords[2 * N + 1] = 0; // dummy
            labels[N] = -1;

            var coordsT = new DenseTensor<float>(coords, new[] { 1, N + 1, 2 });
            var labelsT = new DenseTensor<float>(labels, new[] { 1, N + 1 });
            //Debug.WriteLine("coords:");
            //foreach (var v in coordsT.ToArray()) Debug.Write($"{v} ");
            //Debug.WriteLine("\nlabels:");
            //foreach (var v in labelsT.ToArray()) Debug.Write($"{v} ");
            //Debug.WriteLine("coords and labels");
            return CommonDecoderInputs(embTensor, coordsT, labelsT, origSize);
        }
        // --------------------------------------------------------------------
        // 3.  Build decoder inputs  (box overload)
        // --------------------------------------------------------------------
        public IReadOnlyCollection<NamedOnnxValue> BuildDecoderInputs(
            float[] embedding,
            RectangleF box,
            Size encSize,
            Size origSize)
            => BuildDecoderInputs(
                 embedding,
                 new[] {
                     new PointF(box.Left,  box.Top),
                     new PointF(box.Right, box.Bottom)
                 },
                 encSize,
                 encSize);

        // --------------------------------------------------------------------
        // 4.  Convert decoder output → Gray8 SKBitmap 256×256
        // --------------------------------------------------------------------
        public SKBitmap PostprocessMask(
    IDisposableReadOnlyCollection<DisposableNamedOnnxValue> res,
    float threshold = 0f)
        {
            var maskTensor = res.First(o => o.Name.Contains("masks")).AsTensor<float>();
            var raw = maskTensor.ToArray();
            //Debug.WriteLine("mask----");
            //for (int i = 0; i < 10; i++)
            //    Debug.Write($"{raw[i]} ");
            //Debug.WriteLine("mask----");
            int h = maskTensor.Dimensions[2];
            int w = maskTensor.Dimensions[3];

            // ToArray() because Buffer may not exist in all TFMs
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
            var maskInput = new DenseTensor<float>(new float[1 * 1 * 256 * 256],
                                                   new[] { 1, 1, 256, 256 });
            var hasMask = new DenseTensor<float>(new float[] { 0f }, new[] { 1 });
            var origSizeT = new DenseTensor<float>(
                new float[] { (float)origSize.Height, (float)origSize.Width }, new[] { 2 });

            return new[]
            {
                NamedOnnxValue.CreateFromTensor("image_embeddings", embTensor),
                NamedOnnxValue.CreateFromTensor("point_coords",     coordsTensor),
                NamedOnnxValue.CreateFromTensor("point_labels",     labelsTensor),
                NamedOnnxValue.CreateFromTensor("mask_input",       maskInput),
                NamedOnnxValue.CreateFromTensor("has_mask_input",   hasMask),
                NamedOnnxValue.CreateFromTensor("orig_im_size",     origSizeT)
            };
        }
    }
}
