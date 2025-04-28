// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using SkiaSharp;

namespace PoopDetector.AI.Vision.Processing;

public class ImageProcessingResult
{
    public byte[] Image { get; private set; }
    public string Caption { get; private set; }
    public int InputWidth { get; private set; }
    public int InputHeight { get; private set; }
    public List<BoundingBox> Boxes { get; private set; }
    public List<List<(int x, int y)>> Polygons { get; set; }
    internal ImageProcessingResult(byte[] image=default, string caption = default, List<BoundingBox> boxes=default, int inputWidth=default, int inputHeight=default)
    {
        Image = image;
        Caption = caption;
        Boxes = boxes;
        InputWidth = inputWidth;
        InputHeight = inputHeight;
    }
    internal ImageProcessingResult(byte[] image = default, List<List<(int x, int y)>> polygons = default, int inputWidth = default, int inputHeight = default)
    {
        Image = image;
        Polygons = polygons;
        InputWidth = inputWidth;
        InputHeight = inputHeight;
    }
    public List<SKBitmap> MaskBitmaps { get; set; } = new();
    internal ImageProcessingResult(byte[] image = default, List<SKBitmap> masks = default, int inputWidth = default, int inputHeight = default)
    {
        Image = image;
        InputWidth = inputWidth;
        InputHeight = inputHeight;
        MaskBitmaps = masks;
    }


}