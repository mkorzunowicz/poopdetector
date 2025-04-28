using Microsoft.ML.Data;
using static PoopDetector.AI.IOnnxModel;

namespace PoopDetector.AI;

public class TinyYoloXPoopModel : AbstractOnnxModel
{
    public TinyYoloXPoopModel()
    {
        ColormapList = [new("poop", System.Drawing.Color.Red)];
        ModelInput = "images";
        ModelOutput = "output";
        InputHeight = TinyYoloImageInputData.Height;
        InputWidth = TinyYoloImageInputData.Width;
        Load("yolox_tiny_poop.onnx");
    }

    public TinyYoloXPoopModel(string modelPath)
    {
        ColormapList = [new("poop", System.Drawing.Color.Red)];
        ModelInput = "images";
        ModelOutput = "output";
        InputHeight = TinyYoloImageInputData.Height;
        InputWidth = TinyYoloImageInputData.Width;
        Load(modelPath);
    }

    public TinyYoloXPoopModel(string modelPath, int height, int width)
    {
        ColormapList = [new("poop", System.Drawing.Color.Red)];
        ModelInput = "images";
        ModelOutput = "output";
        InputHeight = height;
        InputWidth = width;
        Load(modelPath);
    }
    public override IImageInputData GetInputData(MLImage image)
    {
        return new TinyYoloImageInputData { Image = image };
    }
}
public class TinyYoloXModel : AbstractOnnxModel
{
    public TinyYoloXModel()
    {
        ColormapList = YoloXColormap.ColormapList;
        ModelInput = "images";
        ModelOutput = "output";
        InputHeight = TinyYoloImageInputData.Height;
        InputWidth = TinyYoloImageInputData.Width;
        Load("yolox_tiny.onnx");
    }

    public override IImageInputData GetInputData(MLImage image)
    {
        return new TinyYoloImageInputData { Image = image };
    }
}

public class NanoYoloXModel : AbstractOnnxModel
{
    public NanoYoloXModel()
    {
        ColormapList = YoloXColormap.ColormapList;
        ModelInput = "images";
        ModelOutput = "output";
        InputHeight = TinyYoloImageInputData.Height;
        InputWidth = TinyYoloImageInputData.Width;
        Load("yolox_nano.onnx");
    }


    public override IImageInputData GetInputData(MLImage image)
    {
        return new TinyYoloImageInputData { Image = image };
    }
}

public class NanoYoloXAndPoopModel : AbstractOnnxModel
{
    public NanoYoloXAndPoopModel()
    {
        ColormapList = YoloXColormap.ColormapList;
        ColormapList.Add(new ("poop", System.Drawing.Color.DodgerBlue));
        ModelInput = "images";
        ModelOutput = "output";
        InputHeight = TinyYoloImageInputData.Height;
        InputWidth = TinyYoloImageInputData.Width;
        Load("yolox_nano_poop_mixed.onnx");
    }

    public override IImageInputData GetInputData(MLImage image)
    {
        return new TinyYoloImageInputData { Image = image };
    }
}

public class SmallYoloXModel : AbstractOnnxModel
{

    public SmallYoloXModel()
    {
        ColormapList = YoloXColormap.ColormapList;
        ModelInput = "images";

        ModelOutput = "output";

        InputHeight = SmallYoloImageInputData.Height;
        InputWidth = SmallYoloImageInputData.Width;
        Load("yolox_s_export_op11.onnx");
        //Load("yolox-s-int8.onnx");
    }

    public override IImageInputData GetInputData(MLImage image)
    {
        return new SmallYoloImageInputData { Image = image };
    }
}