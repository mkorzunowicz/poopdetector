using System;
using System.Drawing;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace PoopDetector.AI;
public class SmallYoloImageInputData : IImageInputData
{
    public const int Height = 640;
    public const int Width = 640;
    [ImageType(Height, Width)]
    public MLImage Image { get; set; }
}

public class TinyYoloImageInputData : IImageInputData
{
    public const int Height = 416;
    public const int Width = 416;
    [ImageType(Height, Width)]
    public MLImage Image { get; set; }
}