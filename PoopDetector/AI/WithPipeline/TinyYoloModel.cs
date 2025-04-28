using Microsoft.ML.Data;
using static PoopDetector.AI.IOnnxModel;

namespace PoopDetector.AI;

public class TinyYoloModel : AbstractOnnxModel
{
    public TinyYoloModel(string modelPath)
    {
        ModelPath = modelPath;
        ColormapList = [
            new("aeroplane", System.Drawing.Color.Aqua),
        new("bicycle", System.Drawing.Color.Red),
        new("bird", System.Drawing.Color.Yellow),
        new("boat", System.Drawing.Color.Aqua),
        new("bottle", System.Drawing.Color.Aqua),
        new("bus", System.Drawing.Color.Aqua),
        new("car", System.Drawing.Color.Aqua),
        new("cat", System.Drawing.Color.Aqua),
        new("chair", System.Drawing.Color.Aqua),
        new("cow", System.Drawing.Color.Aqua),
        new("diningtable", System.Drawing.Color.Aqua),
        new("dog", System.Drawing.Color.Aqua),
        new("horse", System.Drawing.Color.Aqua),
        new("motorbike", System.Drawing.Color.Aqua),
        new("person", System.Drawing.Color.Aqua),
        new("pottedplant", System.Drawing.Color.Aqua),
        new("sheep", System.Drawing.Color.Blue),
        new("sofa", System.Drawing.Color.Aqua),
        new("train", System.Drawing.Color.Aqua),
        new("tvmonitor", System.Drawing.Color.Aqua)];
        ModelInput = "image";
        ModelOutput = "grid";
        InputHeight = TinyYoloImageInputData.Height;
        InputWidth = TinyYoloImageInputData.Width;
        Anchors = [ (1.08f, 1.19f), (3.42f, 4.41f), (6.63f, 11.38f), (9.42f, 5.11f), (16.62f, 10.52f) ];
    }
    public override IImageInputData GetInputData(MLImage image)
    {
        return new TinyYoloImageInputData { Image = image };
    }
}
