// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.Maui.Graphics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;

namespace PoopDetector.AI.Vision.YoloX;

// Reuse of the Ultraface code for YoloX implementation
public class YoloXNanoPoop
    : VisionBase<YoloXNanoImageProcessor>
{
    public const string Identifier = "YoloXNanoPoop";
    public const string ModelFilename = "yolox_nano_poop_cropped_only_best.onnx";
    public const int FeaturesPerBox = 5;
    float nms_threshold = 0.45f;
    float probabilityThreshold = 0.7f;
    List<GridCoordinateAndStride> gridCoords;
    public override Size InputSize => ImageProcessor.RequiredSize;

    public YoloXNanoPoop()
        : base(Identifier, ModelFilename) { }

    protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] image)
    {
        //var st = Stopwatch.StartNew();

        using var preprocessedImage = ImageProcessor.PreprocessSourceImage(image);

        //Debug.WriteLine($"preprocessedImage time: {st.ElapsedMilliseconds}ms");
        //st.Restart();
        // Convert to Tensor of normalized float RGB data with NCHW ordering
        var tensor = ImageProcessor.GetTensorForImage(preprocessedImage);

        //Debug.WriteLine($"tensor time: {st.ElapsedMilliseconds}ms");
        //st.Restart();
        // Run the model
        var predictions = GetPredictions(tensor, preprocessedImage.Width, preprocessedImage.Height);

        //Debug.WriteLine($"predictions time: {st.ElapsedMilliseconds}ms");
        //st.Stop();
        return new ImageProcessingResult(image, null, boxes: predictions, preprocessedImage.Width, preprocessedImage.Height);
    }
    List<BoundingBox> GetPredictions(Tensor<float> input, int inputImageWidth, int inputImageHeight)
    {

        gridCoords ??= YOLOXUtility.GenerateGridCoordinatesWithStrides(YOLOXConstants.Strides, inputImageHeight, inputImageWidth);
        //var st = Stopwatch.StartNew();
        // Setup inputs. Names used must match the input names in the model. 
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", input) };

        // Run inference
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = Session.Run(inputs);

        //Debug.WriteLine($"Prediction time: {st.ElapsedMilliseconds}ms");
        //st.Stop();
        var resultsArray = results.ToArray();
        float[] output = resultsArray[0].AsEnumerable<float>().ToArray();
        var boxesProposals = YOLOXUtility.GenerateBoundingBoxProposals(output, gridCoords, YoloXColormap.PoopList.Count, FeaturesPerBox, probabilityThreshold);

        // Apply Non-Maximum Suppression (NMS) to the proposals
        var bboxes = BBox2DUtility.NMSSortedBoxesOptimized(boxesProposals, nms_threshold);


        var colormap = YoloXColormap.PoopList[0];
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
                Label = colormap.Item1,
                BoxColor = colormap.Item2

            });
        }
        return boxes;
    }
}
