// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;

namespace PoopDetector.AI.Vision.YoloX;

// Reuse of the Ultraface code for YoloX implementation
public class YoloX
    : VisionBase<YoloXImageProcessor>
{
    public const string Identifier = "YoloX";
    public const string ModelFilename = "yolox_s_export_op11.onnx";
    //public const string ModelFilename = "yolox-s-int8.onnx";
    //public const string ModelFilename = "yolox_nano.onnx";

    public YoloX()
        : base(Identifier, ModelFilename) { }

    protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] image)
    {
        // do initial resize maintaining the aspect ratio so the smallest size is 800. this is arbitrary and 
        // chosen to be a good size to dispay to the user with the results
        //using var sourceImage = await Task.Run(() => ImageProcessor.GetImageFromBytes(image, 800f))
        //                                  .ConfigureAwait(false);

        // do the preprocessing to resize the image to the 320x240 with the model expects. 
        // NOTE: this does not maintain the aspect ratio but works well enough with this particular model.
        //       it may be better in other scenarios to resize and crop to convert the original image to a
        //       320x240 image.
        //using var preprocessedImage = await Task.Run(() => ImageProcessor.PreprocessSourceImage(image))
        //                                        .ConfigureAwait(false);

        //// Convert to Tensor of normalized float RGB data with NCHW ordering
        //var tensor = await Task.Run(() => ImageProcessor.GetTensorForImage(preprocessedImage))
        //                       .ConfigureAwait(false);

        //// Run the model
        //var predictions = await Task.Run(() => GetPredictions(tensor, preprocessedImage.Width, preprocessedImage.Height))
        //                            .ConfigureAwait(false);

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
        // Draw the bounding box for the best prediction on the image from the first resize. 
        //byte[] outputImage = default;
        //outputImage = await Task.Run(() => ImageProcessor.ApplyPredictionsToImage(predictions, sourceImage))
        //                            .ConfigureAwait(false);

        return new ImageProcessingResult(image, null, boxes: predictions, preprocessedImage.Width, preprocessedImage.Height);
    }

    List<BoundingBox> GetPredictions(Tensor<float> input, int inputImageWidth, int inputImageHeight)
    {

        var st = Stopwatch.StartNew();
        // Setup inputs. Names used must match the input names in the model. 
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", input) };
        //var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("YOLOX::input_0", input) };

        // Run inference
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = Session.Run(inputs);

        Debug.WriteLine($"Prediction time: {st.ElapsedMilliseconds}ms");
        st.Stop();
        var resultsArray = results.ToArray();
        float[] output = resultsArray[0].AsEnumerable<float>().ToArray();
        float probabilityThreshold = 0.7f;
        var gridCoords = YOLOXUtility.GenerateGridCoordinatesWithStrides(YOLOXConstants.Strides, inputImageHeight, inputImageWidth);
        var boxesProposals = YOLOXUtility.GenerateBoundingBoxProposals(output, gridCoords, YoloXColormap.ColormapList.Count, FeaturesPerBox, probabilityThreshold);
        float nms_threshold = 0.45f;
        // Apply Non-Maximum Suppression (NMS) to the proposals
        List<int> proposal_indices = BBox2DUtility.NMSSortedBoxes(boxesProposals, nms_threshold);

        // Create an array of filtered boxes
        var bboxes = proposal_indices.Select(index => boxesProposals[index]).ToArray();

        var boxes = new List<BoundingBox>();
        foreach (var box in bboxes)
        {
            var colormap = YoloXColormap.ColormapList[box.index];
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
    public const int FeaturesPerBox = 5;
}
