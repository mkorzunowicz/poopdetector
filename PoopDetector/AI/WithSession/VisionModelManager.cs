// PoopDetector.AI.Vision/VisionModelManager.cs
// --------------------------------------------------------------
using CommunityToolkit.Mvvm.ComponentModel;
using PoopDetector.Services;
using PoopDetector.AI.Vision.YoloX;
using System.Collections.Concurrent;

namespace PoopDetector.AI.Vision;

public partial class VisionModelManager : ObservableObject
{
    // singleton
    public static VisionModelManager Instance { get; } = new();

    VisionModelManager() { }

    public IVision? CurrentModel { get; private set; }
    public MobileSam.MobileSam? MobileSam { get; private set; }

    // --------------  progress binding properties  -------------- //
    [ObservableProperty] double _downloadProgress;  // 0-1
    [ObservableProperty] bool _isDownloading;

    // --------------  public API  ------------------------------- //
    public async Task ChangeModelAsync(ModelTypes type,
                                       CancellationToken cancel = default)
    {
        if (CurrentModel is not null &&
            _cache.TryGetValue(type, out var ready) &&
            ready == CurrentModel)
            return;      // already active

        IsDownloading = true;
        DownloadProgress = 0;

        try
        {
            if (type == ModelTypes.FastVLM05B)
            {
                string dec = await EnsureFastVlmInt8Async(new Progress<double>(p => DownloadProgress = p), cancel);
                CurrentModel = new FastVLM.FastVlm(
                    // Prefer fp16 encoder to avoid ConvInteger kernel issues
                    Path.Combine(FileSystem.Current.AppDataDirectory, "vision_encoder_fp16.onnx"),
                    Path.Combine(FileSystem.Current.AppDataDirectory, "embed_tokens_int8.onnx"),
                    Path.Combine(FileSystem.Current.AppDataDirectory, "decoder_model_merged_int8.onnx"),
                    Path.Combine(FileSystem.Current.AppDataDirectory, "vocab.json"),
                    Path.Combine(FileSystem.Current.AppDataDirectory, "merges.txt"),
                    Path.Combine(FileSystem.Current.AppDataDirectory, "tokenizer.json"));
                _cache[type] = CurrentModel;
            }
            else
            {
                string localPath = await EnsureModelFileAsync(type, cancel);
                CurrentModel = CreateVisionWrapper(type, localPath);
                _cache[type] = CurrentModel;
            }
        }
        catch (Exception ex)
        {
            // fallback: stay without a model but keep the app alive
            RaiseError($"Initial model download failed:\n{ex.Message}");
            IsDownloading = false;
            return;
        }
        finally
        {
            IsDownloading = false;
        }
    }
    static async Task<string> EnsureFastVlmInt8Async(IProgress<double> progress, CancellationToken ct)
    {
        // Download 4 artifacts and advance progress by quarters
        double baseProg = 0;
        void ReportPart(double part) => progress.Report(baseProg + 0.25 * part);
        string root = "https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX/resolve/main/";
        _ = await ModelCache.GetAsync(root + "onnx/vision_encoder_fp16.onnx", "vision_encoder_fp16.onnx", new Progress<double>(ReportPart), ct);
        baseProg = 0.25;
        _ = await ModelCache.GetAsync(root + "onnx/embed_tokens_int8.onnx", "embed_tokens_int8.onnx", new Progress<double>(ReportPart), ct);
        baseProg = 0.50;
        string dec = await ModelCache.GetAsync(root + "onnx/decoder_model_merged_int8.onnx", "decoder_model_merged_int8.onnx", new Progress<double>(ReportPart), ct);
        baseProg = 0.75;
        _ = await ModelCache.GetAsync(root + "merges.txt", "merges.txt", new Progress<double>(ReportPart), ct);
        _ = await ModelCache.GetAsync(root + "vocab.json", "vocab.json", new Progress<double>(ReportPart), ct);
        _ = await ModelCache.GetAsync(root + "tokenizer.json", "tokenizer.json", new Progress<double>(ReportPart), ct);
        progress.Report(1.0);
        return dec; // return any; wrapper will resolve others from AppDataDirectory
    }
    const string _defaultUrl =
    "https://github.com/mkorzunowicz/poop_models/raw/refs/heads/main/" +
    "yolox_nano_poop_cropped_only_best.onnx";

    bool _bootstrapped;

    public async Task EnsureDefaultModelAsync()
    {
        MobileSam = new MobileSam.MobileSam();
        if (_bootstrapped || CurrentModel is not null) return;

        // file name cached in AppData
        string name = Path.GetFileName(new Uri(_defaultUrl).AbsolutePath);
        string localPath = Path.Combine(FileSystem.Current.AppDataDirectory, name);

        if (!File.Exists(localPath))    // first app launch
        {
            IsDownloading = true;
            DownloadProgress = 0;
            try
            {
                localPath = await ModelCache.GetAsync(
                _defaultUrl,
                name,
                new Progress<double>(p => DownloadProgress = p));
            }
            catch (Exception ex)
            {
                // fallback: stay without a model but keep the app alive
                RaiseError($"Initial model download failed:\n{ex.Message}");
                IsDownloading = false;
                return;
            }
            IsDownloading = false;
        }

        CurrentModel = new YoloX.YoloX(
            localPath,
            416, 416,
            YoloXColormap.PoopList);

        _bootstrapped = true;
    }
    // --------------  internals  -------------------------------- //
    readonly ConcurrentDictionary<ModelTypes, IVision> _cache = new();

    static async Task<string> EnsureModelFileAsync(ModelTypes t,
                                                   CancellationToken ct)
    {
        var p = new Progress<double>(d =>
            Instance.DownloadProgress = d);     // pushes into binding

        return t switch
        {
            ModelTypes.YoloxNanoPoop =>
                await ModelCache.GetAsync(
                    "https://github.com/mkorzunowicz/poop_models/raw/refs/heads/main/yolox_nano_poop_cropped_only_best.onnx",
                    "yolox_nano_poop_cropped_only_best.onnx", p, ct),

            ModelTypes.YoloxNano =>
                await ModelCache.GetAsync(
                    "https://huggingface.co/yourbucket/yolox_nano.onnx",
                    "yolox_nano.onnx", p, ct),

            ModelTypes.Yolov9ScatSpotter =>
                await ModelCache.GetAsync(
                    // IPFS gateway, CDN URL, S3… – pick one
                    "https://huggingface.co/erotemic/shitspotter-models/resolve/main/models/yolo-v9/shitspotter-simple-v3-run-v06-epoch%3D0032-step%3D000132-trainlosstrain_loss%3D7.603.onnx",
                    "yolov9_poop.onnx", p, ct),

            _ => throw new ArgumentOutOfRangeException()
        };
    }
    public enum Backend
    {
        YoloX,
        Yolov9
    }

    /// <summary>
    /// Download an ONNX from <paramref name="url"/> (once), create the requested
    /// backend wrapper, and make it the <see cref="CurrentModel"/>.
    /// </summary>
    /// <param name="url">HTTP / HTTPS / IPFS gateway link</param>
    /// <param name="backend">Which post-processor to use</param>
    /// <param name="inputW">Model’s expected width  (default 640)</param>
    /// <param name="inputH">Model’s expected height (default 640)</param>
    /// <param name="labels">Class list / colour map</param>
    public async Task LoadRemoteModelAsync(
        string url,
        Backend backend,
        int inputW,
        int inputH,
        List<(string, System.Drawing.Color)> labels,
        CancellationToken cancel = default)
    {
        IsDownloading = true;
        DownloadProgress = 0;

        try
        {
            string localPath = await ModelCache.GetAsync(
                                   url,
                                   Path.GetFileName(new Uri(url).AbsolutePath),
                                   new Progress<double>(p => DownloadProgress = p),
                                   cancel);

            CurrentModel = backend switch
            {
                Backend.YoloX => new YoloX.YoloX(localPath, inputW, inputH, labels),
                Backend.Yolov9 => new Yolov9.Yolov9(localPath, labels),
                _ => throw new ArgumentOutOfRangeException(nameof(backend))
            };
        }
        catch (Exception ex)
        {
            RaiseError($"Could not download model:\n{ex.Message}");
            throw;                                   // still fail if nobody handled
        }
        finally
        {
            IsDownloading = false;
        }
    }
    static IVision CreateVisionWrapper(ModelTypes t, string modelPath)
    {
        var poopAndYolo = YoloXColormap.PoopList.Concat(
                              YoloXColormap.ColormapList).ToList();

        return t switch
        {
            ModelTypes.YoloxNanoPoop =>
                new YoloX.YoloX(modelPath, 416, 416, YoloXColormap.PoopList),

            ModelTypes.YoloxNano =>
                new YoloX.YoloX(modelPath, 416, 416, YoloXColormap.ColormapList),

            ModelTypes.Yolov9ScatSpotter =>
                new Yolov9.Yolov9(modelPath, YoloXColormap.PoopList),

            _ => throw new ArgumentOutOfRangeException()
        };
    }

    // Simple helper to route VLM prompts when the active model supports it
    public async Task<string> AnalyzeWithVlmAsync(byte[] image, string prompt, int maxNewTokens = 12, CancellationToken cancel = default)
    {
        if (CurrentModel is FastVLM.IVisualLanguageModel vlm)
            return await vlm.GenerateAsync(image, prompt, maxNewTokens, cancel);
        throw new InvalidOperationException("Current model is not a VLM");
    }

    /// <summary>
    /// Raised whenever downloading or reading a model fails.
    /// </summary>
    public event EventHandler<string>? DownloadError;

    void RaiseError(string msg)
    {
        DownloadError?.Invoke(this, msg);
#if DEBUG
        System.Diagnostics.Debug.WriteLine("Model download error: " + msg);
#endif
    }
    // ----------------------------------------------------------------- //
    public enum ModelTypes
    {
        YoloxNanoPoop,
        Yolov9ScatSpotter,
        YoloxNano,
        FastVLM05B,
    }
}
