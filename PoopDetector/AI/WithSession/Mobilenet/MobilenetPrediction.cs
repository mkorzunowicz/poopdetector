// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace PoopDetector.AI.Vision.Mobilenet;

public class MobilenetPrediction
{
    public string Label { get; set; }
    public float Confidence { get; set; }
}