// ------------------------------------------------------------
//  ISamModel.cs
// ------------------------------------------------------------

using System.Drawing;
using SkiaSharp;
using PointF = Microsoft.Maui.Graphics.PointF;
using Size = Microsoft.Maui.Graphics.Size;

namespace PoopDetector.AI.Vision;

/// <summary>
/// Common interface for SAM (Segment Anything Model) implementations
/// </summary>
public interface ISamModel
{
    /// <summary>
    /// Encodes the image and prepares for segmentation
    /// </summary>
    Task EncodeAsync(byte[] jpegOrPng);

    /// <summary>
    /// Decodes the encoded image with point prompts
    /// </summary>
    SKBitmap DecodeWithPoints(IReadOnlyList<PointF> points, float threshold = 0f);

    /// <summary>
    /// Decodes the encoded image with a bounding box prompt
    /// </summary>
    SKBitmap DecodeWithBox(RectangleF box, float threshold = 0f);

    /// <summary>
    /// Indicates whether the model is ready to decode (has an encoded image)
    /// </summary>
    bool CanDecode { get; }

    /// <summary>
    /// Gets the encoder size for coordinate mapping
    /// </summary>
    Size GetEncoderSize(Size originalSize);
}
