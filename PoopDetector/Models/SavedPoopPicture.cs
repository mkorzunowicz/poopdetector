// File: Models/SavedPoopPicture.cs
using System;

namespace PoopDetector.Models;

/// <summary>
/// Represents one frozen picture (JPG) together with its
/// SAM mask (transparent PNG) stored on disk.
/// </summary>
public class SavedPoopPicture
{
    public string Id { get; init; }              // same id for jpg / png pair
    public string ImagePath { get; init; }       // …/id.jpg
    public string MaskPath { get; init; }       // …/id_mask.png
    public DateTime CreatedUtc { get; init; }
}
