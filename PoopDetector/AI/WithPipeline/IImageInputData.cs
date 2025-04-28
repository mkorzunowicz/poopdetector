using Microsoft.ML.Data;

namespace PoopDetector.AI
{
    public interface IImageInputData
    {
        MLImage Image { get; set; }
        //byte[] Image { get; set; }
    }
}