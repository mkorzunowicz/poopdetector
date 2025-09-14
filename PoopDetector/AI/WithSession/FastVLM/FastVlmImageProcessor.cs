using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;
using SkiaSharp;

namespace PoopDetector.AI.Vision.FastVLM;

/// Image pre-processing for FastVLM vision encoder.
/// FIXED: Uses FastVLM normalization and 1024x1024 default size matching Rust implementation.
public sealed class FastVlmImageProcessor : SkiaSharpImageProcessor<string, float>
{
    readonly int _size;
    public FastVlmImageProcessor(int size = 1024) => _size = size; // FIXED: Default 1024 to match Rust

    protected override SKBitmap OnPreprocessSourceImage(SKBitmap sourceImage)
    {
        // Resize so the shortest edge equals target, then center-crop to _size x _size
        float ratio = _size / (float)Math.Min(sourceImage.Width, sourceImage.Height);
        using var scaled = sourceImage.Resize(new SKImageInfo((int)Math.Ceiling(sourceImage.Width * ratio),
                                                             (int)Math.Ceiling(sourceImage.Height * ratio)),
                                             SKFilterQuality.Medium);
        var x = Math.Max(0, (scaled.Width - _size) / 2);
        var y = Math.Max(0, (scaled.Height - _size) / 2);
        var rect = SKRectI.Create(x, y, _size, _size);
        using var img = SKImage.FromBitmap(scaled);
        using var cropped = img.Subset(rect);
        return SKBitmap.FromImage(cropped);
    }

    protected override Tensor<float> OnGetTensorForImage(SKBitmap image)
    {
        // FIXED: Use FastVLM normalization matching Rust implementation
        // rescale_factor: 1/255, mean: [0,0,0], std: [1,1,1]
        var rescaleFactor = 1.0f / 255.0f;  // 0.00392156862745098
        var mean = new[] { 0.0f, 0.0f, 0.0f };
        var std = new[] { 1.0f, 1.0f, 1.0f };

        var t = new DenseTensor<float>(new[] { 1, 3, _size, _size });
        for (int y = 0; y < _size; y++)
        {
            for (int x = 0; x < _size; x++)
            {
                var p = image.GetPixel(x, y);
                float r = p.Red * rescaleFactor;  // Scale to [0,1]
                float g = p.Green * rescaleFactor;
                float b = p.Blue * rescaleFactor;
                t[0, 0, y, x] = (r - mean[0]) / std[0];  // = r since mean=0, std=1
                t[0, 1, y, x] = (g - mean[1]) / std[1];  // = g since mean=0, std=1
                t[0, 2, y, x] = (b - mean[2]) / std[2];  // = b since mean=0, std=1
            }
        }
        return t;
    }
}
