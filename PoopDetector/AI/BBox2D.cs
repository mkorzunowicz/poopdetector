
namespace PoopDetector.AI
{
    /// <summary>
    /// A struct that represents a 2D bounding box.
    /// Thanks to https://github.com/cj-mills/unity-bounding-box-2d-toolkit
    /// It also does a proper calculation of the Bounding Boxes. Regular YOLO implementation is different.
    /// </summary>
    public struct BBox2D
    {
        public float x0;
        public float y0;
        public float width;
        public float height;
        public int index;
        public float prob;

        /// <summary>
        /// Initializes a new instance of the BBox2D struct.
        /// </summary>
        /// <param name="x0">The x-coordinate of the top-left corner.</param>
        /// <param name="y0">The y-coordinate of the top-left corner.</param>
        /// <param name="width">The width of the bounding box.</param>
        /// <param name="height">The height of the bounding box.</param>
        /// <param name="index">The class index of the object.</param>
        /// <param name="prob">The probability of the object belonging to the given class.</param>
        public BBox2D(float x0, float y0, float width, float height, int index, float prob)
        {
            this.x0 = x0;
            this.y0 = y0;
            this.width = width;
            this.height = height;
            this.index = index;
            this.prob = prob;
        }
    }

    public static class BBox2DUtility
    {
        /// <summary>
        /// Calculates the union area between two bounding boxes.
        /// </summary>
        /// <param name="a">The first bounding box.</param>
        /// <param name="b">The second bounding box.</param>
        /// <returns>The union area between the two bounding boxes.</returns>
        public static float CalcUnionArea(BBox2D a, BBox2D b)
        {
            // Calculate the coordinates and dimensions of the union area
            float x = MathF.Min(a.x0, b.x0);
            float y = MathF.Min(a.y0, b.y0);
            float w = MathF.Max(a.x0 + a.width, b.x0 + b.width) - x;
            float h = MathF.Max(a.y0 + a.height, b.y0 + b.height) - y;

            // Calculate the union area of two bounding boxes
            return w * h;
        }

        /// <summary>
        /// Calculates the intersection area between two bounding boxes.
        /// </summary>
        /// <param name="a">The first bounding box.</param>
        /// <param name="b">The second bounding box.</param>
        /// <returns>The intersection area between the two bounding boxes.</returns>
        public static float CalcInterArea(BBox2D a, BBox2D b)
        {
            // Calculate the coordinates and dimensions of the intersection area
            float x = MathF.Max(a.x0, b.x0);
            float y = MathF.Max(a.y0, b.y0);
            float w = MathF.Min(a.x0 + a.width, b.x0 + b.width) - x;
            float h = MathF.Min(a.y0 + a.height, b.y0 + b.height) - y;

            // Calculate the intersection area of two bounding boxes
            if (w < 0 || h < 0) return 0;  // No intersection
            return w * h;
        }
        public static List<BBox2D> NMSSortedBoxesOptimized(List<BBox2D> proposals, float nms_thresh = 0.45f)
        {
            List<BBox2D> retainedProposals = new List<BBox2D>(proposals.Count / 2); // Presume half might be filtered out

            for (int i = 0; i < proposals.Count; i++)
            {
                bool keep = true;
                BBox2D a = proposals[i];
                foreach (BBox2D b in retainedProposals)
                {
                    float inter_area = CalcInterArea(a, b);
                    if (inter_area > 0)
                    {
                        float union_area = CalcUnionArea(a, b);
                        if (inter_area / union_area > nms_thresh)
                        {
                            keep = false;
                            break;
                        }
                    }
                }

                if (keep)
                {
                    retainedProposals.Add(a); // Add the actual BBox2D object instead of its index
                }
            }

            return retainedProposals;
        }
        /// <summary>
        /// Performs Non-Maximum Suppression (NMS) on a sorted list of bounding box proposals.
        /// </summary>
        /// <param name="proposals">A sorted list of BBox2D objects representing the bounding box proposals.</param>
        /// <param name="nms_thresh">The NMS threshold for filtering proposals (default is 0.45).</param>
        /// <returns>A list of integers representing the indices of the retained proposals.</returns>
        public static List<int> NMSSortedBoxes(List<BBox2D> proposals, float nms_thresh = 0.45f)
        {
            // Iterate through the proposals and perform non-maximum suppression
            List<int> proposal_indices = new List<int>();

            for (int i = 0; i < proposals.Count; i++)
            {
                // Calculate the intersection and union areas
                BBox2D a = proposals[i];
                bool keep = proposal_indices.All(j =>
                {
                    BBox2D b = proposals[j];
                    float inter_area = CalcInterArea(a, b);
                    float union_area = CalcUnionArea(a, b);
                    // Keep the proposal if its IoU with all previous proposals is below the NMS threshold
                    return inter_area / union_area <= nms_thresh;
                });

                // If the proposal passes the NMS check, add its index to the list
                if (keep) proposal_indices.Add(i);
            }

            return proposal_indices;
        }
    }
}