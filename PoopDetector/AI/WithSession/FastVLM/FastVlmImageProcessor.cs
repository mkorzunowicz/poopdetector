using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;
using SkiaSharp;

namespace PoopDetector.AI.Vision.FastVLM;

/// Image pre-processing for FastVLM vision encoder.
/// Assumes CLIP-style normalization and a square crop (default 448).
public sealed class FastVlmImageProcessor : SkiaSharpImageProcessor<string, float>
{
    readonly int _size;
    public FastVlmImageProcessor(int size = 448) => _size = size;

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
        // CLIP mean/std (OpenAI) – commonly used by VLM encoders
        // If FastVLM expects slightly different stats, this still works reasonably well.
        var mean = new[] { 0.48145466f, 0.4578275f, 0.40821073f };
        var std = new[] { 0.26862954f, 0.26130258f, 0.27577711f };

        var t = new DenseTensor<float>(new[] { 1, 3, _size, _size });
        for (int y = 0; y < _size; y++)
        {
            for (int x = 0; x < _size; x++)
            {
                var p = image.GetPixel(x, y);
                float r = p.Red / 255f;
                float g = p.Green / 255f;
                float b = p.Blue / 255f;
                t[0, 0, y, x] = (r - mean[0]) / std[0];
                t[0, 1, y, x] = (g - mean[1]) / std[1];
                t[0, 2, y, x] = (b - mean[2]) / std[2];
            }
        }
        return t;
    }
}
