// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;

namespace PoopDetector.AI.Vision.YoloX;

// Reuse of the Ultraface code for YoloX implementation
public class YoloXDoubleNanoPoop
    : DoubleVisionBase<YoloXImageProcessor>
{
    public const string Identifier = "YoloXNano";
    public const string ModelFilename = "yolox_nano.onnx";
    public const string Identifier2 = "YoloXNanoPoop";
    public const string ModelFilename2 = "yolox_nano_poop_cropped_only_best.onnx";
    public const int FeaturesPerBox = 5;
    public override Size InputSize => ImageProcessor.RequiredSize;
    public YoloXDoubleNanoPoop()
        : base(Identifier, ModelFilename, Identifier2, ModelFilename2) { }

    protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] image)
    {
        await InitializeAsync().ConfigureAwait(false);
        var st = Stopwatch.StartNew();

        using var preprocessedImage = ImageProcessor.PreprocessSourceImage(image);

        Debug.WriteLine($"preprocessedImage time: {st.ElapsedMilliseconds}ms");
        st.Restart();
        // Convert to Tensor of normalized float RGB data with NCHW ordering
        var tensor = ImageProcessor.GetTensorForImage(preprocessedImage);

        Debug.WriteLine($"tensor time: {st.ElapsedMilliseconds}ms");
        st.Restart();
        // Run the model
        var predictions = GetPredictions(tensor, preprocessedImage.Width, preprocessedImage.Height);

        Debug.WriteLine($"predictions time: {st.ElapsedMilliseconds}ms");
        st.Stop();
        return new ImageProcessingResult(image, null, boxes: predictions);
    }
    List<GridCoordinateAndStride> gridCoords;

    float probabilityThreshold = 0.7f;
    float nms_threshold = 0.45f;

    private List<BoundingBox> GetYoloxPrediction(InferenceSession session, List<NamedOnnxValue> inputs, List<(string, System.Drawing.Color)> colorMap)
    {
        var st = Stopwatch.StartNew();

        // Run inference
        using var results = session.Run(inputs);

        Debug.WriteLine($"Inference time: {st.ElapsedMilliseconds}ms");
        st.Stop();
        float[] output = results.ToArray()[0].AsEnumerable<float>().ToArray();
        var boxesProposals = YOLOXUtility.GenerateBoundingBoxProposals(output, gridCoords, colorMap.Count, FeaturesPerBox, probabilityThreshold);

        // Apply Non-Maximum Suppression (NMS) to the proposals
        var bboxes = BBox2DUtility.NMSSortedBoxesOptimized(boxesProposals, nms_threshold);

        var boxes = new List<BoundingBox>(bboxes.Count);
        foreach (var box in bboxes)
        {
            boxes.Add(new BoundingBox
            {
                Dimensions = new BoundingBoxDimensions
                {
                    X = box.x0,
                    Y = box.y0,
                    Height = box.height,
                    Width = box.width,
                },

                Confidence = box.prob,
                Label = colorMap[box.index].Item1,
                BoxColor = colorMap[box.index].Item2

            });
        }
        Debug.WriteLine($"inference bbox preperation time: {st.ElapsedMilliseconds}ms");
        st.Restart();
        return boxes;
    }

    List<BoundingBox> GetPredictions(Tensor<float> input, int sourceImageWidth, int sourceImageHeight)
    {
        gridCoords ??= YOLOXUtility.GenerateGridCoordinatesWithStrides(YOLOXConstants.Strides, sourceImageHeight, sourceImageWidth);
        // Setup inputs. Names used must match the input names in the model. 
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", input) };

        var yoloBoxes = GetYoloxPrediction(Session, inputs, YoloXColormap.ColormapList);
        var poopBoxes = GetYoloxPrediction(Session2, inputs, YoloXColormap.PoopList);
        yoloBoxes.AddRange(poopBoxes);
        return yoloBoxes;
    }
}
