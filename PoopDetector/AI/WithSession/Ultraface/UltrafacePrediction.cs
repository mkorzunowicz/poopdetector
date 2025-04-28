// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using PoopDetector.AI.Vision.Processing;

namespace PoopDetector.AI.Vision.Ultraface;

public class UltrafacePrediction
{
    public PredictionBox Box { get; set; }
    public float Confidence { get; set; }
    public BoundingBox ToBoundingBox()
    {
        Random random = new Random();
        return new BoundingBox
        {
            BoxColor = System.Drawing.Color.FromArgb((byte)random.Next(0, 255), (byte)random.Next(0, 255), (byte)random.Next(0, 255)),
            Confidence = this.Confidence,
            Dimensions = new BoundingBoxDimensions
            {
                X = Box.Xmin,
                Y = Box.Ymin,
                Height = Box.Xmax - Box.Xmin,
                Width = Box.Ymax - Box.Ymin,
                IsScaled = false
            },
            Label = "face"
        };
    }
}