// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.Maui.Graphics;
using SkiaSharp;

namespace PoopDetector.AI.Vision;

internal static class Utils
{
    internal static async Task<byte[]> LoadResource(string nameOrPath)
    {
        // 1.  Is it an absolute or relative *file-system* path?
        //     (that’s where ModelCache stored the downloaded model)
        if (File.Exists(nameOrPath))
            return await File.ReadAllBytesAsync(nameOrPath);

        // 2.  Otherwise treat it as an *embedded* file inside the .apk / .ipa
        using Stream pkg = await FileSystem.Current.OpenAppPackageFileAsync(nameOrPath);
        using var ms = new MemoryStream();
        await pkg.CopyToAsync(ms);
        return ms.ToArray();
    }
    internal static async Task<bool> PackageResourceAvailable(string nameOrPath)
    {
        try
        {
            using Stream pkg = await FileSystem.Current.OpenAppPackageFileAsync(nameOrPath);
            return pkg.CanRead;
        }
        catch
        {
            return false;
        }
    }


    static async Task<byte[]> GetBytesFromPhotoFile(FileResult fileResult)
    {
        byte[] bytes;

        using Stream stream = await fileResult.OpenReadAsync();
        using MemoryStream ms = new MemoryStream();

        stream.CopyTo(ms);
        bytes = ms.ToArray();

        return bytes;
    }
    public static byte[] HandleOrientation(byte[] image)
    {
        using var memoryStream = new MemoryStream(image);
        using var imageData = SKData.Create(memoryStream);
        using var codec = SKCodec.Create(imageData);
        var orientation = codec.EncodedOrigin;

        using var bitmap = SKBitmap.Decode(image);
        using var adjustedBitmap = AdjustBitmapByOrientation(bitmap, orientation);

        // encode the raw bytes in a known format that SKBitmap.Decode can handle.
        // doing this makes our APIs a little more flexible as they can take multiple image formats as byte[].
        // alternatively we could use SKBitmap instead of byte[] to pass the data around and avoid some
        // SKBitmap.Encode/Decode calls, at the cost of being tightly coupled to the SKBitmap type.
        using var stream = new MemoryStream();
        using var wstream = new SKManagedWStream(stream);

        adjustedBitmap.Encode(wstream, SKEncodedImageFormat.Jpeg, 100);
        var bytes = stream.ToArray();

        return bytes;
    }

    static SKBitmap AdjustBitmapByOrientation(SKBitmap bitmap, SKEncodedOrigin orientation)
    {
        switch (orientation)
        {
            case SKEncodedOrigin.BottomRight:

                using (var canvas = new SKCanvas(bitmap))
                {
                    canvas.RotateDegrees(180, bitmap.Width / 2, bitmap.Height / 2);
                    canvas.DrawBitmap(bitmap.Copy(), 0, 0);
                }

                return bitmap;

            case SKEncodedOrigin.RightTop:

                using (var rotatedBitmap = new SKBitmap(bitmap.Height, bitmap.Width))
                {
                    using (var canvas = new SKCanvas(rotatedBitmap))
                    {
                        canvas.Translate(rotatedBitmap.Width, 0);
                        canvas.RotateDegrees(90);
                        canvas.DrawBitmap(bitmap, 0, 0);
                    }

                    rotatedBitmap.CopyTo(bitmap);
                    return bitmap;
                }

            case SKEncodedOrigin.LeftBottom:

                using (var rotatedBitmap = new SKBitmap(bitmap.Height, bitmap.Width))
                {
                    using (var canvas = new SKCanvas(rotatedBitmap))
                    {
                        canvas.Translate(0, rotatedBitmap.Height);
                        canvas.RotateDegrees(270);
                        canvas.DrawBitmap(bitmap, 0, 0);
                    }

                    rotatedBitmap.CopyTo(bitmap);
                    return bitmap;
                }

            default:
                return bitmap;
        }
    }
    // ――― entry point ―――———————————————————————————————————————————————
    /// <summary>
    /// Convert a Mobile-SAM mask (256×256 Gray8) into a list of COCO-style
    /// polygons (each polygon is a flat list  [x0,y0,x1,y1,…]).
    /// </summary>
    /// <param name="mask">binary mask *already cropped* to usefulW×usefulH</param>
    /// <param name="origW">original frame width  (before padding / resize)</param>
    /// <param name="origH">original frame height (before padding / resize)</param>
    /// <returns>List&lt;float[]&gt; – each float[] is a polygon</returns>
    public static List<float[]> MaskToCocoPolygons(SKBitmap mask,
                                                   int origW, int origH)
    {
        var polys = new List<float[]>();

        int w = mask.Width;
        int h = mask.Height;
        var pix = mask.Bytes;               // Gray8 bytes

        bool[,] visited = new bool[h, w];

        // 4-direction flood-fill to find connected components, then trace outer
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                if (pix[y * w + x] == 0 || visited[y, x]) continue;

                // - start flood for this blob
                var blob = new List<Point>();
                var q = new Queue<Point>();
                q.Enqueue(new Point(x, y));
                visited[y, x] = true;

                while (q.Count > 0)
                {
                    var p = q.Dequeue();
                    blob.Add(p);

                    foreach (var nb in Neigh4(p))
                    {
                        if (nb.X < 0 || nb.X >= w || nb.Y < 0 || nb.Y >= h)
                            continue;
                        if (visited[(int)nb.Y, (int)nb.X] || pix[(int)(nb.Y * w + nb.X)] == 0)
                            continue;

                        visited[(int)nb.Y, (int)nb.X] = true;
                        q.Enqueue(nb);
                    }
                }

                // trace outer contour of the blob (simple Graham scan for convex hull
                // is usually enough for stool-shaped blobs – replace by more precise
                // marching-squares if you need full detail)
                var hull = ConvexHull(blob);

                // map hull coords → original image space
                var seg = new float[hull.Count * 2];
                for (int i = 0; i < hull.Count; i++)
                {
                    seg[2 * i + 0] = (float)hull[i].X / (float)w * origW;
                    seg[2 * i + 1] = (float)hull[i].Y / (float)h * origH;
                }
                polys.Add(seg);
            }

        return polys;
    }

    // ――― helpers ―――———————————————————————————————————————————————
    private static IEnumerable<Point> Neigh4(Point p) => new[]
    {
        new Point(p.X+1,p.Y), new Point(p.X-1,p.Y),
        new Point(p.X,p.Y+1), new Point(p.X,p.Y-1)
    };

    /// <summary>Very small monotone set ⇒ Graham scan convex hull.</summary>
    private static List<Point> ConvexHull(List<Point> pts)
    {
        pts.Sort((a, b) => a.X == b.X ? a.Y.CompareTo(b.Y) : a.X.CompareTo(b.X));
        List<Point> H = new();

        foreach (var p in pts)
        {
            while (H.Count >= 2 && Cross(H[^2], H[^1], p) <= 0) H.RemoveAt(H.Count - 1);
            H.Add(p);
        }
        int t = H.Count + 1;
        for (int i = pts.Count - 1; i >= 0; i--)
        {
            var p = pts[i];
            while (H.Count >= t && Cross(H[^2], H[^1], p) <= 0) H.RemoveAt(H.Count - 1);
            H.Add(p);
        }
        H.RemoveAt(H.Count - 1);
        return H;
    }
    private static int Cross(Point o, Point a, Point b) =>
     (int)  ( (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X));
}


