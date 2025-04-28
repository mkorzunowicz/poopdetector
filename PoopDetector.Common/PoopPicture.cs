namespace PoopDetector.Common
{
    public class PoopPicture
    {
        public byte[] File { get; set; }
        public DateTime DateTime { get; set; }
        public string Geolocation { get; set; }
        public string UserId { get; set; }
        public SubmissionType SubmissionType { get; set; }
        // "Pending", "Approved", "Rejected"
        public string Status { get; set; }
        // [{ "x": 100, "y": 150, "w": 50, "h": 75, "c": 0.7 }]
        public string BoundingBoxes { get; set; }
        public static string EnumToString<T>(T enumValue) where T : Enum
        {
            return enumValue.ToString();
        }
    }

    public enum SubmissionType { BeforeCleanup, AfterCleanup }
}
