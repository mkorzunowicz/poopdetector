namespace PoopDetector.AI;

public class YoloXOutputParser(IOnnxModel onnxModel)
{
    private int InputHeight = onnxModel.InputHeight;
    private int InputWidth = onnxModel.InputWidth;

    // The number of features contained within a box (x, y, height, width, confidence).
    public const int FeaturesPerBox = 5;

    // Labels corresponding to the classes the onnx model can predict.
    private readonly List<(string, System.Drawing.Color)> ColormapList = onnxModel.ColormapList;
    private List<GridCoordinateAndStride> gridCoords;
    public List<BoundingBox> ParseOutputs(float[] modelOutput, float probabilityThreshold = 0.7f)
    {
        gridCoords ??= YOLOXUtility.GenerateGridCoordinatesWithStrides(YOLOXConstants.Strides, InputHeight, InputWidth);
        var boxesProposals = YOLOXUtility.GenerateBoundingBoxProposals(modelOutput, gridCoords, ColormapList.Count, FeaturesPerBox, probabilityThreshold);
        float nms_threshold = 0.45f;
        // Apply Non-Maximum Suppression (NMS) to the proposals
        List<int> proposal_indices = BBox2DUtility.NMSSortedBoxes(boxesProposals, nms_threshold);

        // Create an array of filtered boxes
        var bboxes = proposal_indices.Select(index => boxesProposals[index]).ToArray();

        var boxes = new List<BoundingBox>();
        foreach (var box in bboxes)
        {
            var colormap = ColormapList[box.index];
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