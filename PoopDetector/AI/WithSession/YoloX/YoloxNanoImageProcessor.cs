// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.Maui.Graphics;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;
using SkiaSharp;

namespace PoopDetector.AI.Vision.YoloX;

public class YoloXNanoImageProcessor : SkiaSharpImageProcessor<YoloXPrediction, float>
{
    // Maybe it could be a parameter to somehow fetch this size from API as well
    const int RequiredWidth = 416;
    const int RequiredHeight = 416;
    public override Size RequiredSize => new Size(RequiredWidth, RequiredHeight);

    protected override SKBitmap OnPreprocessSourceImage(SKBitmap sourceImage)
        => sourceImage.Resize(new SKImageInfo(RequiredWidth, RequiredHeight), SKFilterQuality.Medium);

    // For Yolox, the expected input would be 416 x 416 x 4 (in RGBA format)
    readonly int expectedInputLength = RequiredWidth * RequiredHeight * 4;
    readonly int expectedOutputLength = RequiredWidth * RequiredHeight * 3;
    protected override Tensor<float> OnGetTensorForImage(SKBitmap image)
    {
        var bytes = image.GetPixelSpan();

        if (bytes.Length != expectedInputLength)
        {
            throw new Exception($"The parameter {nameof(image)} is an unexpected length. " +
                                $"Expected length is {expectedInputLength}");
        }

        // For the Tensor, we need 3 channels so 416 x 416 x 3 (in RGB format)
        // The channelData array is expected to be in RGB order without a mean applied as opposed to Ultraface
        float[] channelData = new float[expectedOutputLength];

        // Optimized with ChatGPT and seems faster by half on Android
        int pixelCount = RequiredWidth * RequiredHeight;
        for (int i = 0, j = 0; i < pixelCount; i++, j += 4)
        {
            channelData[i] = bytes[j];           // R
            channelData[i + pixelCount] = bytes[j + 1]; // G
            channelData[i + 2 * pixelCount] = bytes[j + 2]; // B
        }

        // NCHW
        return new DenseTensor<float>(channelData, new[] { 1, 3, RequiredHeight, RequiredWidth });
    }

}