namespace PoopDetector.AI.Vision.MobileSam
{
    public class MobileSamPrediction
    {
        /// <summary>
        /// Raw mask array [1×1×1024×1024], or possibly 2D [H×W].
        /// You could store polygons or boundingContours if you convert it.
        /// </summary>
        public float[] MaskData { get; set; }

        // In some workflows, you might store a binary mask, a list of polygons, etc.
        // public bool[] BinaryMask { get; set; }
        public List<List<(int x, int y)>> Polygons { get; set; }
    }
}
