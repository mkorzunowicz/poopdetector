using System.Collections.Generic;

namespace PoopDetector.AI.Vision.Processing
{
    public static class ContourFinder
    {
        public static List<List<(int x, int y)>> FindContours(bool[] mask, int width, int height)
        {
            var visited = new bool[mask.Length];
            var polygons = new List<List<(int x, int y)>>();

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int idx = y * width + x;
                    if (mask[idx] && !visited[idx])
                    {
                        // BFS for region
                        var region = BfsRegion(mask, visited, width, height, x, y);
                        // Extract boundary
                        var boundary = ExtractBoundary(region, mask, width, height);
                        if (boundary.Count > 2) polygons.Add(boundary);
                    }
                }
            }
            return polygons;
        }

        private static List<(int x, int y)> BfsRegion(
            bool[] mask,
            bool[] visited,
            int width,
            int height,
            int sx,
            int sy)
        {
            var queue = new Queue<(int x, int y)>();
            var region = new List<(int x, int y)>();

            queue.Enqueue((sx, sy));
            visited[sy * width + sx] = true;

            var neighbors = new (int dx, int dy)[] {
                (1,0), (-1,0), (0,1), (0,-1),
                (1,1), (1,-1), (-1,1), (-1,-1)
            };

            while (queue.Count > 0)
            {
                var (cx, cy) = queue.Dequeue();
                region.Add((cx, cy));

                foreach (var (dx, dy) in neighbors)
                {
                    int nx = cx + dx, ny = cy + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                    {
                        int nidx = ny * width + nx;
                        if (mask[nidx] && !visited[nidx])
                        {
                            visited[nidx] = true;
                            queue.Enqueue((nx, ny));
                        }
                    }
                }
            }

            return region;
        }

        private static List<(int x, int y)> ExtractBoundary(
            List<(int x, int y)> regionPixels,
            bool[] mask,
            int width,
            int height)
        {
            var boundary = new List<(int x, int y)>();
            var neighbors = new (int dx, int dy)[] {
                (1,0), (-1,0), (0,1), (0,-1),
                (1,1), (1,-1), (-1,1), (-1,-1)
            };

            foreach (var (rx, ry) in regionPixels)
            {
                bool isBoundary = false;
                foreach (var (dx, dy) in neighbors)
                {
                    int nx = rx + dx, ny = ry + dy;
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                    {
                        isBoundary = true;
                        break;
                    }
                    int nidx = ny * width + nx;
                    if (!mask[nidx])
                    {
                        isBoundary = true;
                        break;
                    }
                }
                if (isBoundary) boundary.Add((rx, ry));
            }

            return boundary;
        }
    }
}
