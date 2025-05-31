// Copyright © 2025
// MIT License

using Microsoft.Maui.Graphics;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;
using PoopDetector.AI.Vision.YoloX;
using SkiaSharp;

namespace PoopDetector.AI.Vision.Yolov9;

/// <summary>
/// Minimal pre-processor for the YOLO-v9 model.
/// Resizes the bitmap to the exporter’s 640 × 640 input and packs
/// raw 0-255 **BGR** bytes into an NCHW float tensor.
/// </summary>
public sealed class Yolov9ImageProcessor
    : SkiaSharpImageProcessor<YoloXPrediction, float>   // reuse common DTO
{
    int _w = 640, _h = 640;

    public void Configure(int w, int h) { _w = w; _h = h; }

    public override Size RequiredSize => new Size(_w, _h);

    protected override SKBitmap OnPreprocessSourceImage(SKBitmap src) =>
        src.Resize(new SKImageInfo(_w, _h), SKFilterQuality.Medium);

    protected override Tensor<float> OnGetTensorForImage(SKBitmap img)
    {
        ReadOnlySpan<byte> src = img.GetPixelSpan();
        int pix = _w * _h;

        float[] data = new float[pix * 3];

        for (int i = 0, j = 0; i < pix; ++i, j += 4)
        {
            //   ONNX exported by kwcoco/yolov9 expects BGR
            data[i]          = src[j + 2];   // B  (R channel in SK bitmap)
            data[i + pix]    = src[j + 1];   // G
            data[i + 2*pix]  = src[j];       // R  (B channel in SK bitmap)
        }

        return new DenseTensor<float>(data,
                                      new[] { 1, 3, _h, _w });
    }
}
