namespace PoopDetector.AI
{
    public class YOLOXConstants
    {
        // Stride values used by the YOLOX model
        public static readonly int[] Strides = [8, 16, 32];

        // Number of fields in each bounding box
        public static readonly int NumBBoxFields = 5;
    }
    /// <summary>
    /// A struct for grid coordinates and stride information.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the GridCoordinateAndStride struct.
    /// </remarks>
    /// <param name="xCoordinate">The x-coordinate of the grid.</param>
    /// <param name="yCoordinate">The y-coordinate of the grid.</param>
    /// <param name="stride">The stride value for the grid.</param>
    public struct GridCoordinateAndStride(int xCoordinate, int yCoordinate, int stride)
    {
        public int xCoordinate = xCoordinate;
        public int yCoordinate = yCoordinate;
        public int stride = stride;
    }

    /// <summary>
    /// Utility class for YOLOX-related operations.
    /// </summary>
    public static class YOLOXUtility
    {
        /// <summary>
        /// Generates a list of GridCoordinateAndStride objects based on input strides, height, and width.
        /// </summary>
        /// <param name="strides">An array of stride values.</param>
        /// <param name="height">The height of the grid.</param>
        /// <param name="width">The width of the grid.</param>
        /// <returns>A list of GridCoordinateAndStride objects.</returns>
        public static List<GridCoordinateAndStride> GenerateGridCoordinatesWithStrides(int[] strides, int height, int width)
        {
            // Generate a list of GridCoordinateAndStride objects by iterating through possible grid positions and strides
            return strides.SelectMany(stride => Enumerable.Range(0, height / stride)
                                                           .SelectMany(g1 => Enumerable.Range(0, width / stride)
                                                                                        .Select(g0 => new GridCoordinateAndStride(g0, g1, stride)))).ToList();
        }

        /// <summary>
        /// Generates a list of bounding box proposals based on the model output, grid strides, and other parameters.
        /// </summary>
        /// <param name="modelOutput">The output of the YOLOX model.</param>
        /// <param name="gridCoordsAndStrides">A list of GridCoordinateAndStride objects.</param>
        /// <param name="numClasses">The number of object classes.</param>
        /// <param name="numBBoxFields">The number of bounding box fields.</param>
        /// <param name="confidenceThreshold">The confidence threshold for filtering proposals.</param>
        /// <returns>A list of BBox2D objects representing the generated proposals.</returns>
        public static List<BBox2D> GenerateBoundingBoxProposals(float[] modelOutput, List<GridCoordinateAndStride> gridCoordsAndStrides, int numClasses, int numBBoxFields, float confidenceThreshold)
        {
            int proposalLength = numClasses + numBBoxFields;

            // Process the model output to generate a list of BBox2D objects
            return gridCoordsAndStrides.Select((grid, anchorIndex) =>
            {
                int startIndex = anchorIndex * proposalLength;

                // Calculate coordinates and dimensions of the bounding box
                float centerX = (modelOutput[startIndex] + grid.xCoordinate) * grid.stride;
                float centerY = (modelOutput[startIndex + 1] + grid.yCoordinate) * grid.stride;
                float w = MathF.Exp(modelOutput[startIndex + 2]) * grid.stride;
                float h = MathF.Exp(modelOutput[startIndex + 3]) * grid.stride;

                // Initialize BBox2D object
                BBox2D obj = new(
                    centerX - w * 0.5f,
                    centerY - h * 0.5f,
                    w, h, 0, 0);

                // Compute objectness and class probabilities for each bounding box
                float box_objectness = modelOutput[startIndex + 4];

                for (int classIndex = 0; classIndex < numClasses; classIndex++)
                {
                    float boxClassScore = modelOutput[startIndex + numBBoxFields + classIndex];
                    float boxProb = box_objectness * boxClassScore;

                    // Update the object with the highest probability and class label
                    if (boxProb > obj.prob)
                    {
                        obj.index = classIndex;
                        obj.prob = boxProb;
                    }
                }

                return obj;
            })
            .Where(obj => obj.prob > confidenceThreshold) // Filter by confidence threshold
            .OrderByDescending(x => x.prob) // Sort by probability
            .ToList();
        }
    }
}