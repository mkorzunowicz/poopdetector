using SkiaSharp;

namespace PoopDetector.AI.Vision.MobileSam
{
    /// <summary>
    /// Resizes an image so that the long side is 1024, preserving aspect ratio,
    /// then zero-pads the other dimension to 1024.
    /// Returns a float[1,3,1024,1024] for the encoder, plus the actual resizedW/resizedH.
    /// </summary>
    public static class AspectRatioResizer
    {
        public const int TargetLongSide = 1024;

        public static (float[] tensor, int resizedW, int resizedH)
            ResizeAndPadTo1024(SKBitmap original, float[] mean, float[] std)
        {
            int origW = original.Width;
            int origH = original.Height;

            // 1) Figure out which side is the "long side"
            int resizedW = origW;
            int resizedH = origH;
            if (origW > origH)
            {
                // landscape => width = 1024
                resizedW = TargetLongSide;
                // keep aspect ratio
                float ratio = (float)TargetLongSide / (float)origW;
                resizedH = (int)(ratio * origH);
            }
            else
            {
                // portrait => height = 1024
                resizedH = TargetLongSide;
                float ratio = (float)TargetLongSide / (float)origH;
                resizedW = (int)(ratio * origW);
            }

            // 2) Resize the image to (resizedW x resizedH)
            var resized = new SKBitmap(resizedW, resizedH);
            original.ScalePixels(resized, SKFilterQuality.Medium);

            // 3) Convert to float array of shape (1,3,resizedH,resizedW) w/ mean/std
            float[] partialData = MakeCHWTensor(resized, mean, std);

            // 4) If resizedW < 1024 or resizedH < 1024, we pad.
            // final shape => (1,3,1024,1024)
            float[] finalData = new float[1 * 3 * TargetLongSide * TargetLongSide];

            // We'll copy the partial data into the top-left region
            // partial shape => [1,3,resizedH,resizedW]
            // We'll store it line by line in the final array
            int cStrideInPartial = resizedW * resizedH;           // plane size in partial
            int cStrideInFinal = TargetLongSide * TargetLongSide; // plane size in final

            for (int c = 0; c < 3; c++)
            {
                for (int row = 0; row < resizedH; row++)
                {
                    // partial offset
                    int partialRowStart = c * cStrideInPartial + row * resizedW;
                    // final offset
                    int finalRowStart = c * cStrideInFinal + row * TargetLongSide;
                    // copy 'resizedW' elements
                    System.Array.Copy(
                        partialData,
                        partialRowStart,
                        finalData,
                        finalRowStart,
                        resizedW
                    );
                }
            }

            return (finalData, resizedW, resizedH);
        }

        /// <summary>
        /// Converts an SKBitmap (resized) to float[1,3,H,W], channel-first, with mean/std.
        /// </summary>
        private static float[] MakeCHWTensor(SKBitmap bmp, float[] mean, float[] std)
        {
            int w = bmp.Width;
            int h = bmp.Height;
            int c = 3;

            float[] data = new float[c * w * h];
            var pixels = bmp.Pixels; // BGRA or RGBA in memory, depends on platform

            int hw = w * h;
            for (int i = 0; i < hw; i++)
            {
                uint rgba = (uint)pixels[i];
                byte b = (byte)((rgba >> 0) & 0xFF);
                byte g = (byte)((rgba >> 8) & 0xFF);
                byte r = (byte)((rgba >> 16) & 0xFF);
                // alpha (rgba >> 24)...

                float fr = (r - mean[0]) / std[0];
                float fg = (g - mean[1]) / std[1];
                float fb = (b - mean[2]) / std[2];

                data[0 * hw + i] = fr; // R
                data[1 * hw + i] = fg; // G
                data[2 * hw + i] = fb; // B
            }

            return data;
        }
    }
}
