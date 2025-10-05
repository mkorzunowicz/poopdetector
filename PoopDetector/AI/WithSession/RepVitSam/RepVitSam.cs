// ------------------------------------------------------------
//  RepVitSam.cs
// ------------------------------------------------------------

using System.Drawing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;
using SkiaSharp;

using PointF = Microsoft.Maui.Graphics.PointF;
using Size = Microsoft.Maui.Graphics.Size;

namespace PoopDetector.AI.Vision.RepVitSam
{
    /// <summary>
    /// Two-stage RepViT-SAM wrapper:
    ///   • EncodeAsync(byte[])  → runs the encoder and stores embeddings
    ///   • DecodeWithPoints / DecodeWithBox → runs the decoder on those embeddings
    ///   
    /// Key differences from MobileSAM:
    ///   • Fixed 1024×1024 input size (no adaptive scaling)
    ///   • Input tensor name is "x" instead of default
    ///   • Based on RepViT architecture from THU-MIG/RepViT
    /// </summary>
    public sealed class RepVitSam
        : DoubleVisionBase<RepVitSamImageProcessor>, IDisposable, ISamModel
    {
        //private const string EncoderOnnx = "repvit_sam_image_encoder.onnx"; //~1700-1900 // a bit more accurate
        //private const string DecoderOnnx = "repvit_sam_image_decoder.onnx";
        //private const string EncoderOnnx = "repvit_sam_encoder_quantized.onnx"; // ~ 1700-2000 more on the higher side, seems like the coordinates are not working
        //private const string DecoderOnnx = "repvit_sam_decoder_quantized.onnx"; // Android ~~7000-9000, still something off with the coordinates, but works if it will hit the right spot

        private const string EncoderOnnx = "repvit_sam_encoder.onnx"; //~1600-2000
        private const string DecoderOnnx = "repvit_sam_decoder.onnx"; // Android Up to 12000ms and sometimes the BB doesn't hit

        public RepVitSam()
            : base("RepVitSAMEncoder", EncoderOnnx,
                   "RepVitSAMDecoder", DecoderOnnx)
        { }

        // --------------------------------------------------------------------
        // internal state after EncodeAsync
        // --------------------------------------------------------------------
        private float[] _embedding;          // [1,256,64,64]
        private Size _origSize;            // original camera frame
        private const int ImageSize = 1024;  // RepViT-SAM uses fixed 1024×1024

        private const int EmbeddingC = 256;
        private const int EmbeddingH = 64;
        private const int EmbeddingW = 64;
        
        public bool CanDecode => _embedding != null;

        // --------------------------------------------------------------------
        // GetEncoderSize for ISamModel interface
        // Since RepVitSam uses fixed 1024×1024, return that
        // --------------------------------------------------------------------
        public Size GetEncoderSize(Size originalSize) => new Size(ImageSize, ImageSize);

        // --------------------------------------------------------------------
        // Step 1 – Encoder
        // --------------------------------------------------------------------
        public async Task EncodeAsync(byte[] jpegOrPng)
        {
            await InitializeAsync().ConfigureAwait(false);

            using SKBitmap bmp = ImageProcessor.PreprocessSourceImage(jpegOrPng);
            _origSize = new Size(bmp.Width, bmp.Height);

            Tensor<float> imgTensor = ImageProcessor.GetTensorForImage(bmp);

            // run encoder
            using var res = Session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(
                    Session.InputMetadata.Keys.First(), imgTensor)
            });

            _embedding = res.First().AsTensor<float>().ToArray();
        }

        // --------------------------------------------------------------------
        // Step 2A – Decoder with foreground points
        // --------------------------------------------------------------------
        public SKBitmap DecodeWithPoints(IReadOnlyList<PointF> points,
                                         float threshold = 0f)
        {
            if (_embedding == null)
                throw new InvalidOperationException("Call EncodeAsync first.");

            var inputs = ImageProcessor.BuildDecoderInputs(
                _embedding, points, _origSize);

            using var res = Session2.Run(inputs);
            return ImageProcessor.PostprocessMask(res, _origSize, threshold);
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
                _embedding, box, _origSize);

            using var res = Session2.Run(inputs);
            return ImageProcessor.PostprocessMask(res, _origSize, threshold);
        }

        // --------------------------------------------------------------------
        //  we don't use the one-shot OnProcessImageAsync in this model
        // --------------------------------------------------------------------
        protected override Task<ImageProcessingResult>
            OnProcessImageAsync(byte[] image) =>
            throw new NotSupportedException(
                "Use EncodeAsync + DecodeWithPoints/DecodeWithBox instead.");

        public void Dispose()
        {
            Session?.Dispose();
            Session2?.Dispose();
        }
    }
}
