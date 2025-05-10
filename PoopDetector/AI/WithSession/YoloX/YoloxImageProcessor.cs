// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Microsoft.Maui.Graphics;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;
using SkiaSharp;

namespace PoopDetector.AI.Vision.YoloX;

/// <summary>
///  Image-processor whose input size can be configured at runtime.
///  Configuration *must* be done once, before the first inference,
///  via <see cref="Configure(int,int)"/>.
/// </summary>
public sealed class YoloXImageProcessor
    : SkiaSharpImageProcessor<YoloXPrediction, float>
{
    
     int _width  = 416;
     int _height = 416;

    /// <summary>Call this *once* from the model ctor.</summary>
    public void Configure(int width, int height)
    {
        _width  = width;
        _height = height;
    }

    public override Size RequiredSize => new Size(_width, _height);

    protected override SKBitmap OnPreprocessSourceImage(SKBitmap sourceImage) =>
        sourceImage.Resize(new SKImageInfo(_width, _height), SKFilterQuality.Medium);

    protected override Tensor<float> OnGetTensorForImage(SKBitmap image)
    {
        var bytes = image.GetPixelSpan();
        int pixelCount   = _width * _height;

        if (bytes.Length != pixelCount * 4)
            throw new InvalidOperationException(
                $"Image buffer has unexpected length {bytes.Length}.");

        float[] data = new float[pixelCount * 3];

        for (int i = 0, j = 0; i < pixelCount; i++, j += 4)
        {
            data[i]                 = bytes[j];       // R
            data[i + pixelCount]    = bytes[j + 1];   // G
            data[i + 2 * pixelCount] = bytes[j + 2];  // B
        }

        return new DenseTensor<float>(data, new[] { 1, 3, _height, _width });
    }
}
