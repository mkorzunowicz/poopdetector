// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PoopDetector.AI.Vision.Processing;

namespace PoopDetector.AI.Vision.YoloX;

// Reuse of the Ultraface code for YoloX implementation
public class YoloXInt8
    : VisionBase<YoloXInt8ImageProcessor>
{
    public const string Identifier = "YoloXInt8";
    //public const string ModelFilename = "yolox_s_export_op11.onnx";
    public const string ModelFilename = "yolox-s-int8.onnx";
    //public const string ModelFilename = "yolox_nano.onnx";

    public YoloXInt8()
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


        using var preprocessedImage = ImageProcessor.PreprocessSourceImage(image);

        // Convert to Tensor of normalized float RGB data with NCHW ordering
        var tensor = ImageProcessor.GetTensorForImage(preprocessedImage);

        // Run the model
        var predictions = GetPredictionsYoloInt8(tensor, preprocessedImage.Width, preprocessedImage.Height);

        // Draw the bounding box for the best prediction on the image from the first resize. 
        //byte[] outputImage = default;
        //outputImage = await Task.Run(() => ImageProcessor.ApplyPredictionsToImage(predictions, sourceImage))
        //                            .ConfigureAwait(false);

        return new ImageProcessingResult(image, null, boxes: predictions);
    }

    List<BoundingBox> GetPredictionsYoloInt8( Tensor<float> input, int sourceImageWidth, int sourceImageHeight)
    {
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("YOLOX::input_0", input) };

        // Run inference
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = Session.Run(inputs);
        // Run the inference
        
        // Assuming there are three outputs as described
        var outputs = results.ToList();

        var boundingBoxes = new List<BoundingBox>();
        // Decode and process each output tensor
        foreach (var result in outputs)
        {
            var tensor = result.AsTensor<float>();
            long numDetections = tensor.Length / 85;  // Assuming each detection has 85 elements
            
            for (int i = 0; i < numDetections; i++)
            {
                int startIndex = i * 85;
                var dimensions = new BoundingBoxDimensions
                {
                    X = tensor[startIndex],
                    Y = tensor[startIndex + 1],
                    Width = tensor[startIndex + 2],
                    Height = tensor[startIndex + 3]
                };
                //TODO: still needs a loop here over classes
                var colormap = YoloXColormap.ColormapList[0];
                var confidence = tensor[startIndex + 4];
                if (confidence > 0.5)  // Apply confidence threshold
                {
                    var boundingBox = new BoundingBox
                    {
                        Dimensions = dimensions,
                        Label = colormap.Item1,  // This should be set based on class scores which start at startIndex + 5
                        Confidence = confidence,
                        BoxColor = colormap.Item2  // Just an example, you might set the color based on the object type or other criteria
                    };
                    boundingBoxes.Add(boundingBox);
                }
            }
        }
        return boundingBoxes;
    }

    public const int FeaturesPerBox = 5;
}
