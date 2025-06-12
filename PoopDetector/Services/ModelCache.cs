// PoopDetector.AI.Vision.Services.ModelCache.cs
// --------------------------------------------------------------
using System.Net;
using System.Security.Cryptography;
using PoopDetector.AI.Vision;

namespace PoopDetector.Services;

/// <summary>
/// Handles download-once / reuse-later logic for ONNX models.
/// </summary>
public static class ModelCache
{
    /// <param name="remoteUrl">Full HTTP(S) or IPFS gateway URL of the model</param>
    /// <param name="fileName">Short local file name ( e.g. "yolov9.onnx" )</param>
    /// <param name="progress">
    ///     Optional progress reporter (0-1).  Forward this into your
    ///     view-model to update a <see cref="ProgressBar"/> or <see cref="ActivityIndicator"/>.
    /// </param>
    /// <returns>Absolute path of the ready-to-use ONNX file on local storage.</returns>
    public static async Task<string> GetAsync(
    string remoteUrl,
    string fileName,
    IProgress<double>? progress = null,
    CancellationToken cancel = default)
    {
        if (await Utils.PackageResourceAvailable(fileName))
            return fileName;

            string dir = FileSystem.Current.AppDataDirectory;
        Directory.CreateDirectory(dir);

        string path = Path.Combine(dir, fileName);

        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(15) };

        // ───────────────────────────────────────────────────────────────
        // 1. Query remote size (HEAD) so we can compare / report resume
        // ───────────────────────────────────────────────────────────────
        long? remoteSize = null;
        try
        {
            using var head = new HttpRequestMessage(HttpMethod.Head, remoteUrl);
            using var resp = await http.SendAsync(head, cancel);
            if (resp.IsSuccessStatusCode)
                remoteSize = resp.Content.Headers.ContentLength;
        }
        catch { /* HEAD may fail, we'll cope later */ }

        long localSize = File.Exists(path) ? new FileInfo(path).Length : 0;

        // remoteSize unknown  →  cannot verify integrity → delete and start fresh
        if (remoteSize.HasValue && localSize == remoteSize.Value && localSize > 0)
        {
            progress?.Report(1);
            return path;                       // already complete
        }
        if (remoteSize == null || localSize > remoteSize)
        {
            TryDelete(path);
            localSize = 0;
        }

        // ───────────────────────────────────────────────────────────────
        // 2. Start or resume download
        // ───────────────────────────────────────────────────────────────
        var request = new HttpRequestMessage(HttpMethod.Get, remoteUrl);

        if (localSize > 0)                    // attempt resume
            request.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(localSize, null);

        using var response = await http.SendAsync(
                                 request,
                                 HttpCompletionOption.ResponseHeadersRead,
                                 cancel);

        if (!(response.IsSuccessStatusCode ||
              response.StatusCode == HttpStatusCode.PartialContent))
            throw new HttpRequestException(
                  $"Server returned {(int)response.StatusCode} – {response.ReasonPhrase}");

        long totalLength = remoteSize ??
                           (response.Content.Headers.ContentLength + localSize
                            ?? -1);

        // open stream (append if resuming)
        await using FileStream dst = new(
            path,
            localSize > 0 ? FileMode.Append : FileMode.Create,
            FileAccess.Write,
            FileShare.None);

        await using Stream src = await response.Content.ReadAsStreamAsync(cancel);

        byte[] buffer = new byte[81920];
        long readTotal = localSize;
        int read;

        void Report()              // local inline helper
        {
            if (totalLength > 0)
                progress?.Report(readTotal / (double)totalLength);
        }
        Report();                  // show resume offset immediately

        while ((read = await src.ReadAsync(buffer, cancel)) > 0)
        {
            await dst.WriteAsync(buffer.AsMemory(0, read), cancel);
            readTotal += read;
            Report();
        }
        progress?.Report(1);

        // final size sanity check
        if (remoteSize.HasValue && readTotal != remoteSize.Value)
            throw new IOException("Downloaded size mismatch (file may be corrupt).");

        return path;
    }
    // ───────────────────────────────────────────────────────────────────────────
    static void TryDelete(string p) { try { if (File.Exists(p)) File.Delete(p); } catch { } }
}
