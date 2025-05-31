// Copyright © 2025
// MIT License – same as the rest of the project.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;

namespace PoopDetector.AI.Vision.Yolov9;

/// <summary>
/// Stand-alone wrapper for the “shitspotter” YOLO-v9 ONNX export
/// (one input, nine output tensors – DFL head + PGI head).
/// Implements the same public surface as <see cref="YoloX.YoloX"/>.
/// </summary>
public sealed class Yolov9 : VisionBase<Yolov9ImageProcessor>
{
    // -----------------  fixed model constants  --------------------- //
    const int IMG_W = 640;
    const int IMG_H = 640;
    const int REG_MAX = 16;                // DFL bins

    // Only the *final* (PGI-B) head is used for inference
    static readonly (string Reg, string Cls, string Obj)[] HEAD =
    {
        ("2981", "2986", "2957"),  // 80×80  – stride 8
        ("3028", "3033", "3004"),  // 40×40  – stride 16
        ("3075", "3080", "3051"),  // 20×20  – stride 32
    };
    static readonly int[] STRIDES = { 8, 16, 32 };

    // --------------------  user-tweakable  ------------------------- //
    readonly List<(string Label, System.Drawing.Color Color)> _colormap;
    readonly float _confThresh;
    readonly float _nmsThresh;

    // Re-usable scratch buffers
    readonly float[] _expectBuf = new float[REG_MAX];

    public Yolov9(string onnxFile,
                  List<(string, System.Drawing.Color)> colormap,
                  float confThresh = 0.40f,
                  float nmsThresh  = 0.45f)
        : base("Yolov9", onnxFile)
    {
        _colormap   = colormap ?? throw new ArgumentNullException(nameof(colormap));
        _confThresh = confThresh;
        _nmsThresh  = nmsThresh;

        if (ImageProcessor is Yolov9ImageProcessor p)
            p.Configure(IMG_W, IMG_H);
    }

    public override Size InputSize => new Size(IMG_W, IMG_H);

    // --------------------------  inference  ------------------------ //
    protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] image)
    {
        using var pre   = ImageProcessor.PreprocessSourceImage(image);
        var inputTensor = ImageProcessor.GetTensorForImage(pre);

        var boxes = Decode(Session.Run(
                               new[] { NamedOnnxValue.CreateFromTensor("input", inputTensor) }),
                           pre.Width, pre.Height);

        return new ImageProcessingResult(image, null, boxes,
                                         pre.Width, pre.Height);
    }

    // -------------------  post-processing  ------------------------- //
    List<BoundingBox> Decode(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> ortOut,
                             int srcW, int srcH)
    {
        var cand = new List<Candidate>(4096);
        int C = _colormap.Count;

        for (int scale = 0; scale < HEAD.Length; ++scale)
        {
            var reg = ortOut.First(o => o.Name == HEAD[scale].Reg).AsTensor<float>();
            var cls = ortOut.First(o => o.Name == HEAD[scale].Cls).AsTensor<float>();
            var obj = ortOut.First(o => o.Name == HEAD[scale].Obj).AsTensor<float>();

            int S = STRIDES[scale];
            int H = reg.Dimensions[2];
            int W = reg.Dimensions[3];

            for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
            {
                float objectness = Sigmoid(obj[0, y, x]);
                if (objectness < 1e-4f) continue;

                // ---- DFL expectation for all 4 sides ---- //
                float l = Expect(reg, 0, y, x) * S;
                float t = Expect(reg, 1, y, x) * S;
                float r = Expect(reg, 2, y, x) * S;
                float b = Expect(reg, 3, y, x) * S;

                float cx = (x + 0.5f) * S;
                float cy = (y + 0.5f) * S;

                float x0 = MathF.Max(0, cx - l);
                float y0 = MathF.Max(0, cy - t);
                float x1 = MathF.Min(srcW - 1, cx + r);
                float y1 = MathF.Min(srcH - 1, cy + b);

                for (int c = 0; c < C; ++c)
                {
                    float conf = objectness * Sigmoid(cls[c, y, x]);
                    if (conf < _confThresh) continue;

                    cand.Add(new Candidate
                    {
                        x0 = x0, y0 = y0, x1 = x1, y1 = y1,
                        score = conf,
                        label = c
                    });
                }
            }
        }

        var final = Nms(cand, _nmsThresh);

        // ---- map to public BoundingBox ---- //
        var list = new List<BoundingBox>(final.Count);
        foreach (var b in final)
        {
            var cmap = _colormap[b.label % _colormap.Count];
            list.Add(new BoundingBox
            {
                Dimensions = new BoundingBoxDimensions
                {
                    X      = b.x0,
                    Y      = b.y0,
                    Width  = b.x1 - b.x0,
                    Height = b.y1 - b.y0
                },
                Confidence = b.score,
                Label      = cmap.Item1,
                BoxColor   = cmap.Item2
            });
        }
        return list;
    }

    // -----------------------  helpers  ----------------------------- //
    float Expect(Tensor<float> t, int side, int y, int x)
    {
        int baseCh = 0;                       // 16 bins start at channel 0
        for (int b = 0; b < REG_MAX; ++b)
            _expectBuf[b] = t[baseCh + b, side, y, x];

        SoftmaxInPlace(_expectBuf);
        float exp = 0;
        for (int b = 0; b < REG_MAX; ++b)
            exp += _expectBuf[b] * b;
        return exp;
    }

    static void SoftmaxInPlace(Span<float> v)
    {
        float max = v[0];
        for (int i = 1; i < v.Length; ++i) max = MathF.Max(max, v[i]);

        float sum = 0;
        for (int i = 0; i < v.Length; ++i)
        {
            v[i] = MathF.Exp(v[i] - max);
            sum += v[i];
        }
        for (int i = 0; i < v.Length; ++i) v[i] /= sum;
    }

    static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    // ------------------  class-wise greedy NMS  -------------------- //
    static List<Candidate> Nms(List<Candidate> boxes, float iouThr)
    {
        var sorted = boxes.OrderByDescending(b => b.score).ToList();
        var keep   = new List<Candidate>();

        foreach (var box in sorted)
        {
            bool discard = keep.Any(k => k.label == box.label &&
                                         IoU(k, box) > iouThr);
            if (!discard) keep.Add(box);
        }
        return keep;
    }

    static float IoU(Candidate a, Candidate b)
    {
        float ix0 = MathF.Max(a.x0, b.x0);
        float iy0 = MathF.Max(a.y0, b.y0);
        float ix1 = MathF.Min(a.x1, b.x1);
        float iy1 = MathF.Min(a.y1, b.y1);

        float iw = MathF.Max(0, ix1 - ix0);
        float ih = MathF.Max(0, iy1 - iy0);
        float inter = iw * ih;

        float areaA = (a.x1 - a.x0) * (a.y1 - a.y0);
        float areaB = (b.x1 - b.x0) * (b.y1 - b.y0);

        return inter / (areaA + areaB - inter + 1e-6f);
    }

    struct Candidate
    {
        public float x0, y0, x1, y1;
        public float score;
        public int   label;
    }
}
