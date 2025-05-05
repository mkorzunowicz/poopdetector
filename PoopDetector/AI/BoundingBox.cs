
using System.Drawing;
using SkiaSharp;
using PoopDetector.AI.Vision;
using Color = System.Drawing.Color;
using Size = Microsoft.Maui.Graphics.Size;
using PointF = Microsoft.Maui.Graphics.PointF;
using System.Diagnostics;

namespace PoopDetector.AI;

public class CompundPredictionResult
{
    List<PredictionResult> Results { get; set; }
}
public class PredictionResult
{
    public async Task RunSamEncode()
    {
        var sw = Stopwatch.StartNew();
        await VisionModelManager.Instance.MobileSam.EncodeAsync(InputImage);
        Debug.WriteLine($"Encode took {sw.ElapsedMilliseconds} ms");
    }
    public async void ClearSamPoints()
    {
        PreviousPoints.Clear();
        MaskBitmaps.Clear();
    }
    public async Task RunSamDecode(PointF point = default)
    {
        var sw = Stopwatch.StartNew();
        if (!VisionModelManager.Instance.MobileSam.CanDecode) return;
        SKBitmap mask256;
        if (!point.IsEmpty)
        {
            PreviousPoints.Add(point);
            mask256 = VisionModelManager.Instance.MobileSam.DecodeWithPoints(PreviousPoints);
            Debug.WriteLine($"Decode took {sw.ElapsedMilliseconds} ms");
            sw.Restart();

            MaskBitmaps = [mask256];
            MaskToPolygon();

            Debug.WriteLine($"Mask to polygons took {sw.ElapsedMilliseconds} ms");
            return;
        }
        var enc = VisionModelManager.Instance.MobileSam.ImageProcessor
                      .GetEncoderSize(new Size(OriginalWidth, OriginalHeight));

        if (BoundingBoxes != null && BoundingBoxes.Count > 0)
        {
            var box = BoundingBoxes[0];
            var sx = enc.Width / InputWidth;
            var sy = enc.Height / InputHeight;

            var x_enc = box.Rect.X * sx;
            var y_enc = box.Rect.Y * sy;
            var x2_enc = (box.Rect.X + box.Rect.Width) * sx;
            var y2_enc = (box.Rect.Y + box.Rect.Height) * sy;

            // The box doesn't work unfortunately - not in a meaningful way at least..
            //mask256 = VisionModelManager.Instance.MobileSam
            //            .DecodeWithBox(new RectangleF((float)x_enc, (float)y_enc, (float)x2_enc, (float)y2_enc));

            var xmid = x_enc + (x2_enc - x_enc) / 2;
            var ymid = y_enc + (y2_enc - y_enc) / 2;

            // Therefore we go for a point in the middle of the box
            PreviousPoints = new List<PointF> { new((float)xmid, (float)ymid) };
            mask256 = VisionModelManager.Instance.MobileSam.DecodeWithPoints(PreviousPoints);
            Debug.WriteLine($"Decode took {sw.ElapsedMilliseconds} ms");
        }
        else
        {
            // no box -> single click in the centre
            PreviousPoints = new List<PointF> { new((float)enc.Width / 2f, (float)enc.Height / 2f) };
            //var centre = new List<PointF> { new(OriginalWidth / 2f, OriginalHeight / 2f) };
            mask256 = VisionModelManager.Instance.MobileSam.DecodeWithPoints(PreviousPoints);
            Debug.WriteLine($"Decode took {sw.ElapsedMilliseconds} ms");
        }
        MaskBitmaps = [mask256];

        sw.Restart();
        MaskToPolygon();
        Debug.WriteLine($"Mask to polygons took {sw.ElapsedMilliseconds} ms");

    }
    public List<List<int>> Polygons { get; private set; }
    public string MaskToPolygon()
    {
        var cocoSegs = MaskBitmaps.SelectMany(mb=> Utils.MaskToCocoPolygons(mb,
                                         OriginalWidth,
                                         OriginalHeight));

        // store or serialise:
        Polygons = cocoSegs
            .Select(seg => seg
                .Select((v, i) => i % 2 == 0 ? (int)v : (int)v)   // ints if you prefer
                .ToList())
            .ToList();        

        var json = System.Text.Json.JsonSerializer.Serialize(
              new { segmentation = cocoSegs });
        return json;
    }
    private List<PointF> PreviousPoints { get; set; } = [];
    public byte[] InputImage { get; set; }
    //public MLImage InputImage { get; set; }
    public List<BoundingBox> BoundingBoxes = [];
    //public List<List<(int x, int y)>> Polygons { get; set; } = new();
    public List<SKBitmap> MaskBitmaps { get; set; } = new();
    public int OriginalHeight { get; set; }
    public int OriginalWidth { get; set; }
    public int InputHeight { get; set; }
    public int InputWidth { get; set; }
    private float areaRatio = -1;
    //float minSizeRatio = 0.5f;
    float maxTotalAreaRatio = 0.25f;
    float minTotalAreaRatio = 0.10f;
    public bool Drawn { get; set; }
    private float AreaRatio { get { return TotalCoveredAreaRatio; } }
    //private float OriginalArea => OriginalHeight * OriginalWidth;
    //private float InputArea => InputHeight * InputWidth;
    public DateTime TimeStamp { get; set; }
    public bool IsTooSmall => AreaRatio < minTotalAreaRatio;
    public bool IsGood => !IsTooBig && !IsTooSmall
        ;
    public bool IsTooBig => AreaRatio > maxTotalAreaRatio;
    public float ScaleResizeX => (float)OriginalWidth / (float)InputWidth;
    public float ScaleResizeY => (float)OriginalHeight / (float)InputHeight;
    public float TotalCoveredAreaRatio
    {
        get
        {
            if (BoundingBoxes.Count == 0) return 0;
            if (areaRatio != -1) return areaRatio;

            var inputArea = InputHeight * InputWidth;
            if (BoundingBoxes.Count == 1)
            {
                var boxArea = BoundingBoxes[0].Area;

                areaRatio = boxArea / inputArea;
                return areaRatio;
            }
            if (BoundingBoxes.Count > 1)
            {
                // TODO: find the one that is in the center (has margins away from the sides maybe?)
                // TODO2: if horizonatal poop position in vertical view
                var boxArea = BoundingBoxes[0].Area;

                areaRatio = boxArea / inputArea;
                return areaRatio;
            }
            return 0;

            // Create a bitmap representing the image area
            //using var bitmap = new Bitmap(OriginalWidth, OriginalHeight);
            //using (var graphics = Graphics.FromImage(bitmap))
            //{
            //    graphics.Clear(Color.Black);

            //    foreach (var bbox in BoundingBoxes)
            //    {
            //        graphics.FillRectangle(Brushes.White, bbox.Rect);
            //    }
            //}

            //// Count white pixels to determine the covered area
            //int whitePixelCount = 0;
            //for (int y = 0; y < bitmap.Height; y++)
            //{
            //    for (int x = 0; x < bitmap.Width; x++)
            //    {
            //        if (bitmap.GetPixel(x, y).ToArgb() == Color.White.ToArgb())
            //        {
            //            whitePixelCount++;
            //        }
            //    }
            //}

            //areaRatio = OriginalWidth * OriginalHeight;
            //return areaRatio;
        }
    }


    public string BoundingBoxesToJson()
    {
        return $"[{string.Join(',', BoundingBoxes.Select(b => b.ToJson()))}]";
    }
}
public class BoundingBoxDimensions
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Height { get; set; }
    public float Width { get; set; }
    public float X2 { get { return X + Width; } }
    public float Y2 { get { return Y + Height; } }
    public bool IsScaled { get; set; }
    // { "x": 100, "y": 150, "w": 50, "h": 75, "c": 0.7 }
    public string ToJson() { return $"{{ \"x\": {X}, \"y\": {Y}, \"w\": {Width}, \"h\": {Height} }}"; }
}

public class BoundingBox
{
    public BoundingBoxDimensions Dimensions { get; set; }

    public string Label { get; set; }

    public float Confidence { get; set; }

    public RectangleF Rect
    {
        get { return new RectangleF(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height); }
    }
    public float Area => Dimensions.Height * Dimensions.Width;

    public Color BoxColor { get; set; }

    public string Description => $"{Label} ({(Confidence * 100).ToString("0")}%)";

    // { "x": 100, "y": 150, "w": 50, "h": 75, "c": 0.7 }
    public string ToJson()
    {
        return $"{{ \"x\": {Dimensions.X}, \"y\": {Dimensions.Y}, \"w\": {Dimensions.Width}, \"h\": {Dimensions.Height} }}, \"c\":{Confidence}";
    }
}