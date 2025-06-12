// PoopDetector.AI.Vision.Yolov9/Yolov9.cs
// --------------------------------------------------------------
// Batch-safe decoder for shitspotter YOLO-v9 (DFL + objectness)
// --------------------------------------------------------------

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;

namespace PoopDetector.AI.Vision.Yolov9;

public sealed class Yolov9 : VisionBase<Yolov9ImageProcessor>
{
    const int IMG_W = 640, IMG_H = 640, REG_MAX = 16;
    static readonly (string Reg, string Cls, string Obj)[] HEAD =
    {
        ("2981", "2986", "2957"), // 80×80 – stride 8
        ("3028", "3033", "3004"), // 40×40 – stride 16
        ("3075", "3080", "3051"), // 20×20 – stride 32
    };
    static readonly int[] STRIDES = { 8, 16, 32 };

    readonly List<(string Label, System.Drawing.Color Color)> _cmap;
    readonly float _confThr, _nmsThr;
    readonly float[] _soft = new float[REG_MAX];

    public Yolov9(string onnx,
                  List<(string, System.Drawing.Color)> colormap,
                  float confThr = .40f, float nmsThr = .45f)
        : base("Yolov9", onnx)
    {
        _cmap = colormap;
        _confThr = confThr;
        _nmsThr = nmsThr;
        (ImageProcessor as Yolov9ImageProcessor)!.Configure(IMG_W, IMG_H);
    }

    public override Size InputSize => new Size(IMG_W, IMG_H);

    protected override async Task<ImageProcessingResult>
        OnProcessImageAsync(byte[] img)
    {
        using var pre = ImageProcessor.PreprocessSourceImage(img);
        var inp = ImageProcessor.GetTensorForImage(pre);

        using var ortOut = Session.Run(
            new[] { NamedOnnxValue.CreateFromTensor("input", inp) });

        var boxes = Decode(ortOut, pre.Width, pre.Height);
        return new ImageProcessingResult(img, null, boxes,
                                         pre.Width, pre.Height);
    }

    // ---------------------- decoder ------------------------------ //
    List<BoundingBox> Decode(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> o,
                             int srcW, int srcH)
    {
        var cand = new List<Cand>(4096);
        int C = _cmap.Count;

        for (int s = 0; s < HEAD.Length; ++s)
        {
            var reg = o.First(x => x.Name == HEAD[s].Reg).AsTensor<float>();
            var cls = o.First(x => x.Name == HEAD[s].Cls).AsTensor<float>();
            var obj = o.First(x => x.Name == HEAD[s].Obj).AsTensor<float>();

            int S = STRIDES[s];
            int H = reg.Dimensions[3]; // note: dims = [N,16,4,H,W]
            int W = reg.Dimensions[4];

            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                {
                    float pObj = Sigmoid(obj[0, 0, y, x]);
                    if (pObj < 1e-4f) continue;

                    float l = Expect(reg, 0, y, x) * S;
                    float t = Expect(reg, 1, y, x) * S;
                    float r = Expect(reg, 2, y, x) * S;
                    float b = Expect(reg, 3, y, x) * S;

                    float cx = (x + .5f) * S, cy = (y + .5f) * S;
                    float x0 = MathF.Max(0, cx - l), y0 = MathF.Max(0, cy - t);
                    float x1 = MathF.Min(srcW - 1, cx + r), y1 = MathF.Min(srcH - 1, cy + b);

                    for (int c = 0; c < C; ++c)
                    {
                        float conf = pObj * Sigmoid(cls[0, c, y, x]);
                        if (conf < _confThr) continue;
                        cand.Add(new Cand(x0, y0, x1, y1, conf, c));
                    }
                }
        }

        var fin = Nms(cand, _nmsThr);
        var outL = new List<BoundingBox>(fin.Count);
        foreach (var b in fin)
        {
            var cm = _cmap[b.lbl % _cmap.Count];
            outL.Add(new BoundingBox
            {
                Dimensions = new BoundingBoxDimensions
                {
                    X = b.x0,
                    Y = b.y0,
                    Width = b.x1 - b.x0,
                    Height = b.y1 - b.y0
                },
                Confidence = b.sc,
                Label = cm.Item1,
                BoxColor = cm.Item2
            });
        }
        return outL;
    }

    float Expect(Tensor<float> t, int side, int y, int x)
    {
        for (int i = 0; i < REG_MAX; ++i)
            _soft[i] = t[0, i, side, y, x]; // N,Bin,Side,H,W

        Softmax(_soft); float e = 0;
        for (int i = 0; i < REG_MAX; ++i) e += _soft[i] * i;
        return e;
    }

    static void Softmax(Span<float> v)
    {
        float m = v[0]; for (int i = 1; i < v.Length; ++i) m = MathF.Max(m, v[i]);
        float s = 0; for (int i = 0; i < v.Length; ++i) { v[i] = MathF.Exp(v[i] - m); s += v[i]; }
        for (int i = 0; i < v.Length; ++i) v[i] /= s;
    }
    static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    // -------------------- NMS + helpers -------------------------- //
    static List<Cand> Nms(List<Cand> v, float thr)
    {
        var keep = new List<Cand>();
        foreach (var b in v.OrderByDescending(z => z.sc))
        {
            if (keep.Any(k => k.lbl == b.lbl && IoU(k, b) > thr)) continue;
            keep.Add(b);
        }
        return keep;
    }
    static float IoU(Cand a, Cand b)
    {
        float ix0 = MathF.Max(a.x0, b.x0), iy0 = MathF.Max(a.y0, b.y0);
        float ix1 = MathF.Min(a.x1, b.x1), iy1 = MathF.Min(a.y1, b.y1);
        float iw = MathF.Max(0, ix1 - ix0), ih = MathF.Max(0, iy1 - iy0);
        float inter = iw * ih;
        float areaA = (a.x1 - a.x0) * (a.y1 - a.y0),
              areaB = (b.x1 - b.x0) * (b.y1 - b.y0);
        return inter / (areaA + areaB - inter + 1e-6f);
    }
    readonly record struct Cand(float x0, float y0, float x1, float y1,
                                float sc, int lbl);
}
