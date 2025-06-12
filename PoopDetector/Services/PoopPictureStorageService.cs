// File: Services/PoopPictureStorageService.cs
using Microsoft.Maui.Storage;
using PoopDetector.Models;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace PoopDetector.Services;

/// <summary>
/// Persists accepted pictures (jpg) and their SAM masks (png)
/// in the sandbox so no runtime permission is required.
/// </summary>
public sealed class PoopPictureStorageService
{
    private readonly string _root =
        Path.Combine(FileSystem.Current.AppDataDirectory, "saved_pictures");

    public PoopPictureStorageService() => Directory.CreateDirectory(_root);

    public async Task SaveAsync(byte[] imageBytes, SKBitmap maskBitmap)
    {
        if (imageBytes == null || maskBitmap == null) return;

        var id = Guid.NewGuid().ToString("N");
        var jpg = Path.Combine(_root, $"{id}.jpg");
        var png = Path.Combine(_root, $"{id}_mask.png");

        // save JPG
        await File.WriteAllBytesAsync(jpg, imageBytes);

        // save PNG mask (with alpha)
        using var data = maskBitmap.Encode(SKEncodedImageFormat.Png, 100);
        await File.WriteAllBytesAsync(png, data.ToArray());
    }

    public Task<IReadOnlyList<SavedPoopPicture>> GetAllAsync()
    {
        var pics = Directory.EnumerateFiles(_root, "*.jpg")
            .Select(jpg =>
            {
                var id = Path.GetFileNameWithoutExtension(jpg);
                var png = Path.Combine(_root, $"{id}_mask.png");
                if (!File.Exists(png)) return null;        // ignore orphaned files

                return new SavedPoopPicture
                {
                    Id = id,
                    ImagePath = jpg,
                    MaskPath = png,
                    CreatedUtc = File.GetCreationTimeUtc(jpg)
                };
            })
            .Where(p => p != null)
            .OrderByDescending(p => p.CreatedUtc)          // newest first
            .ToList()
            .AsReadOnly();

        return Task.FromResult((IReadOnlyList<SavedPoopPicture>)pics);
    }
}
