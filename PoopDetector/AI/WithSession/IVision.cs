// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.Maui.Graphics;
using PoopDetector.AI.Vision.Processing;

namespace PoopDetector.AI.Vision;

public interface IVision
{
    string Name { get; }
    string ModelName { get; }
    Size InputSize { get; }
    Task InitializeAsync();
    Task UpdateExecutionProviderAsync(ExecutionProviders executionProvider);
    Task<ImageProcessingResult> ProcessImageAsync(byte[] image);
}
