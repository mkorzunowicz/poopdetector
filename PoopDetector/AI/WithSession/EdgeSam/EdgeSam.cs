// ------------------------------------------------------------
//  EdgeSam.cs - EdgeSAM Implementation
// ------------------------------------------------------------

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;
using SkiaSharp;
using System.Drawing;
using PointF = Microsoft.Maui.Graphics.PointF;
using Size = Microsoft.Maui.Graphics.Size;

namespace PoopDetector.AI.Vision.EdgeSam;

/// <summary>
/// EdgeSAM: Efficient Segment Anything Model
/// Two-stage: encoder -> decoder with prompts
/// </summary>
public sealed class EdgeSam 
    : DoubleVisionBase<EdgeSamImageProcessor>, IDisposable, ISamModel
{
    //private const string EncoderOnnx = "edge_sam_encoder.onnx"; // ~700-1200ms, seems very precise, but the mask is choppy, although if we take a polygon, it hardly matters
    //private const string DecoderOnnx = "edge_sam_decoder.onnx"; // Android 3800-4000ms
    private const string EncoderOnnx = "edge_sam_3x_encoder.onnx"; // ~800-1200ms, seems very precise as well
    private const string DecoderOnnx = "edge_sam_3x_decoder.onnx"; // Android ~3200.. hm after another run it went up to 3800-4100 as the othr model.. why?

    public EdgeSam()
        : base("EdgeSAMEncoder", EncoderOnnx,
               "EdgeSAMDecoder", DecoderOnnx)
    { }

    // --------------------------------------------------------------------
    // State
    // --------------------------------------------------------------------
    private float[]? _embedding;
    private Size _origSize;
    public bool CanDecode => _embedding != null;

    // Fixed encoder input size (1024x1024)
    public const int ImageSize = 1024;

    // --------------------------------------------------------------------
    // ISamModel Implementation
    // --------------------------------------------------------------------
    
    /// <summary>
    /// Gets the encoder size for coordinate mapping
    /// </summary>
    public Size GetEncoderSize(Size originalSize)
    {
        // EdgeSAM uses fixed 1024x1024 encoder size
        return new Size(ImageSize, ImageSize);
    }

    // --------------------------------------------------------------------
    // Step 1 – Encoder: image -> embedding
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

    // --------------------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------------------
    public void Dispose()
    {
        Session?.Dispose();
        Session2?.Dispose();
    }
}
