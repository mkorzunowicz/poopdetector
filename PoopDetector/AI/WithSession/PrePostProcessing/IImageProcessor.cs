﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Collections.Generic;
using Microsoft.Maui.Graphics;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PoopDetector.AI.Vision.Processing;

public interface IImageProcessor<TImage, TPrediction, TTensor>
{
    Size OriginalSize { get; set; }
    // pre-process the image.
    TImage PreprocessSourceImage(byte[] sourceImage);

    // convert the PreprocessSourceImage output to a Tensor of type TTensor 
    Tensor<TTensor> GetTensorForImage(TImage image);

    // apply the predictions to the image if applicable
    // e.g. draw the bounding box around an area selected by the model
    byte[] ApplyPredictionsToImage(IList<TPrediction> predictions, TImage image);

    Size RequiredSize { get; }
}