using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;

namespace PoopDetector.AI.Vision.YoloX;

/// <summary>
///  A *single* YOLOX wrapper that can handle Nano, Tiny, Small …
///  Pass (filename, width, height, numClasses) and you’re done.
/// </summary>
public class YoloX : VisionBase<YoloXImageProcessor>
{
    readonly int _width;
    readonly int _height;
    readonly List<(string, System.Drawing.Color)> _colormap;
    readonly float _probThresh = 0.7f;
    readonly float _nmsThresh = 0.45f;
    const int _featuresPerBox = 5;

    List<GridCoordinateAndStride> _grid;

    /// <summary>
    ///  Create any YOLOX variant in one line.
    /// </summary>
    /// <param name="modelFile">ONNX file embedded as a resource.</param>
    /// <param name="width">Model’s input width.</param>
    /// <param name="height">Model’s input height.</param>
    /// <param name="numClasses">Number of classes in the model output.</param>
    public YoloX(string modelFile, int width, int height, List<(string, System.Drawing.Color)> colormap)
        : base("YoloX", modelFile)
    {
        _width = width;
        _height = height;
        _colormap = colormap;

        if (ImageProcessor is YoloXImageProcessor p)
            p.Configure(width, height);
        else
            throw new InvalidOperationException("Unexpected processor type.");
    }

    public override Size InputSize => new Size(_width, _height);

    // -------------------------------------------------------------------- //
    //                         inference pipeline                           //
    // -------------------------------------------------------------------- //
    protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] image)
    {
        // await InitializeAsync().ConfigureAwait(false);
        using var pre = ImageProcessor.PreprocessSourceImage(image);
        var tensor = ImageProcessor.GetTensorForImage(pre);
        var boxes = GetPredictions(tensor, pre.Width, pre.Height);

        return new ImageProcessingResult(image, null, boxes, pre.Width, pre.Height);
    }

    List<BoundingBox> GetPredictions(Tensor<float> input, int w, int h)
    {
        _grid ??= YOLOXUtility.GenerateGridCoordinatesWithStrides(
                      YOLOXConstants.Strides, h, w);

        using var results = Session.Run(
            new[] { NamedOnnxValue.CreateFromTensor("images", input) });

        float[] output = results.First().AsEnumerable<float>().ToArray();

        var proposals = YOLOXUtility.GenerateBoundingBoxProposals(
                            output, _grid, _colormap.Count,
                            _featuresPerBox, _probThresh);

        var best = BBox2DUtility.NMSSortedBoxesOptimized(proposals, _nmsThresh);

        var list = new List<BoundingBox>(best.Count);

        foreach (var b in best)
        {
            var cmap = _colormap[b.index % _colormap.Count]; // wrap if needed
            list.Add(new BoundingBox
            {
                Dimensions = new BoundingBoxDimensions
                {
                    X = b.x0,
                    Y = b.y0,
                    Width = b.width,
                    Height = b.height
                },
                Confidence = b.prob,
                Label = cmap.Item1,
                BoxColor = cmap.Item2
            });
        }

        return list;
    }
}
