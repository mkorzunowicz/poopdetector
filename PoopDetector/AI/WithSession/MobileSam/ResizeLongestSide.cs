
//using System;
//using SkiaSharp;

//namespace MobileSAM
//{
//    /// <summary>
//    /// Mirrors the ResizeLongestSide logic from Segment-Anything.
//    /// We rely on SkiaSharp for image resizing.
//    /// </summary>
//    public class ResizeLongestSide
//    {
//        private readonly int _targetLength;

//        public ResizeLongestSide(int targetLength)
//        {
//            _targetLength = targetLength;
//        }

//        /// <summary>
//        /// Applies a resize so that the max dimension of 'image' is _targetLength.
//        /// Expects an 8-bit RGB image in a SKBitmap.
//        /// </summary>
//        public SKBitmap ApplyImage(SKBitmap image)
//        {
//            var (newH, newW) = GetPreprocessShape(image.Height, image.Width, _targetLength);

//            // Resize with Skia
//            SKImageInfo info = new SKImageInfo(newW, newH, image.ColorType, image.AlphaType);
//            SKBitmap resized = new SKBitmap(info);
//            image.ScalePixels(resized, SKFilterQuality.Medium);
//            return resized;
//        }

//        /// <summary>
//        /// Applies the same scale transform to coordinates.
//        /// coords shape: [*, 2], last dimension is (x,y).
//        /// originalSize is (H,W).
//        /// </summary>
//        public (float x, float y)[] ApplyCoords((float x, float y)[] coords, (int h, int w) originalSize)
//        {
//            var (newH, newW) = GetPreprocessShape(originalSize.h, originalSize.w, _targetLength);
//            float scaleW = (float)newW / (float)originalSize.w;
//            float scaleH = (float)newH / (float)originalSize.h;

//            var result = new (float x, float y)[coords.Length];
//            for (int i = 0; i < coords.Length; i++)
//            {
//                result[i] = (coords[i].x * scaleW, coords[i].y * scaleH);
//            }
//            return result;
//        }

//        /// <summary>
//        /// Applies the transform to bounding boxes: shape is [*, 4] in XYXY.
//        /// </summary>
//        public (float x1, float y1, float x2, float y2)[] ApplyBoxes(
//            (float x1, float y1, float x2, float y2)[] boxes,
//            (int h, int w) originalSize)
//        {
//            // We can treat each box as two coords: (x1,y1), (x2,y2)
//            // Then re-pack them.
//            var coords = new (float x, float y)[boxes.Length * 2];
//            for (int i = 0; i < boxes.Length; i++)
//            {
//                coords[2 * i] = (boxes[i].x1, boxes[i].y1);
//                coords[2 * i + 1] = (boxes[i].x2, boxes[i].y2);
//            }

//            var scaled = ApplyCoords(coords, originalSize);

//            var result = new (float x1, float y1, float x2, float y2)[boxes.Length];
//            for (int i = 0; i < boxes.Length; i++)
//            {
//                var p1 = scaled[2 * i];
//                var p2 = scaled[2 * i + 1];
//                result[i] = (p1.x, p1.y, p2.x, p2.y);
//            }

//            return result;
//        }

//        /// <summary>
//        /// Computes new (H,W) so that max dimension is targetLength.
//        /// </summary>
//        public static (int newH, int newW) GetPreprocessShape(int oldH, int oldW, int longSideLength)
//        {
//            float scale = (float)longSideLength / Math.Max(oldH, oldW);
//            int newH = (int)MathF.Round(oldH * scale);
//            int newW = (int)MathF.Round(oldW * scale);
//            return (newH, newW);
//        }
//    }
//}
