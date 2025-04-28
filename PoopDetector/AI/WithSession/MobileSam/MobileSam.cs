// ------------------------------------------------------------
//  MobileSam.cs
// ------------------------------------------------------------

using System.Diagnostics;
using System.Drawing;
using Microsoft.Maui.Graphics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;
using SkiaSharp;

using PointF = Microsoft.Maui.Graphics.PointF;
using Size = Microsoft.Maui.Graphics.Size;
namespace PoopDetector.AI.Vision.MobileSam
{
    /// <summary>
    /// Two-stage MobileSAM wrapper:
    ///   • EncodeAsync(byte[])  → runs the encoder and stores embeddings
    ///   • DecodeWithPoints / DecodeWithBox → runs the decoder on those embeddings
    /// </summary>
    public sealed class MobileSam
        : DoubleVisionBase<MobileSamImageProcessor>, IDisposable
    {
        private const string EncoderOnnx = "mobile_sam_image_encoder.onnx";
        //private const string DecoderOnnx = "sam_onnx_decoder.onnx";
        private const string DecoderOnnx = "sam_onnx_quantized_decoder.onnx";

        public MobileSam()
            : base("MobileSAMEncoder", EncoderOnnx,
                   "MobileSAMDecoder", DecoderOnnx)
        { }

        // --------------------------------------------------------------------
        // internal state after EncodeAsync
        // --------------------------------------------------------------------
        private float[]? _embedding;          // [1,256,64,64]
        private Size _origSize;            // original camera frame
        private Size _encSize;             // (w_res,h_res) long-side-1024

        private const int EmbeddingC = 256;
        private const int EmbeddingH = 64;
        private const int EmbeddingW = 64;
        public bool CanDecode => _embedding != null;
        // --------------------------------------------------------------------
        // Step 1 – Encoder
        // --------------------------------------------------------------------
        public async Task EncodeAsync(byte[] jpegOrPng)
        {
            //ImageProcessor.OriginalSize = _origSize = new Size(width, height);
            await InitializeAsync().ConfigureAwait(false);

            using SKBitmap bmp = ImageProcessor.PreprocessSourceImage(jpegOrPng);
            _origSize = new Size(bmp.Width, bmp.Height);
            _encSize = ImageProcessor.GetEncoderSize(_origSize);

            Tensor<float> imgTensor = ImageProcessor.GetTensorForImage(bmp);

            // run encoder
            using var res = Session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(
                    Session.InputMetadata.Keys.First(), imgTensor)
            });

            _embedding = res.First().AsTensor<float>().ToArray();
            //Debug.WriteLine($"emb len {_embedding.Length}");
            //for (int i = 0; i < 10; i++)
            //    Debug.Write($"{_embedding[i]} ");
            //Debug.WriteLine($"[MobileSAM] encoder OK – embedding len {_embedding.Length}");
        }

        // --------------------------------------------------------------------
        // Step 2A – Decoder with foreground points
        // --------------------------------------------------------------------
        public SKBitmap DecodeWithPoints(IReadOnlyList<PointF> points,
                                         float threshold = 0f)
        {
            if (_embedding == null)
                throw new InvalidOperationException("Call EncodeAsync first.");
            _origSize = _encSize;
            var inputs = ImageProcessor.BuildDecoderInputs(
                _embedding, points, _encSize, _encSize);

            using var res = Session2.Run(inputs);
            return ImageProcessor.PostprocessMask(res, threshold);
        }

        // --------------------------------------------------------------------
        // Step 2B – Decoder with bounding box
        // --------------------------------------------------------------------
        public SKBitmap DecodeWithBox(RectangleF box,
                                      float threshold = 0f)
        {
            if (_embedding == null)
                throw new InvalidOperationException("Call EncodeAsync first.");

            var inputs = ImageProcessor.BuildDecoderInputs(
                _embedding, box, _encSize, _origSize);

            using var res = Session2.Run(inputs);
            return ImageProcessor.PostprocessMask(res, threshold);
        }

        // --------------------------------------------------------------------
        //  we don’t use the one-shot OnProcessImageAsync in this model
        // --------------------------------------------------------------------
        protected override Task<ImageProcessingResult>
            OnProcessImageAsync(byte[] image) =>
            throw new NotSupportedException(
                "Use EncodeAsync + DecodeWithPoints/DecodeWithBox instead.");

        public void Dispose()
        {
            Session.Dispose();
            Session2.Dispose();
        }
    }
}
