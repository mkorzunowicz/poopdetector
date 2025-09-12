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
    public async Task ChangeModelAsync(ModelTypes type)
    {
        await ChangeModelAsync(type, "", CancellationToken.None);
    }

    public async Task ChangeModelAsync(ModelTypes type, CancellationToken cancel)
    {
        await ChangeModelAsync(type, "", cancel);
    }

    public async Task ChangeModelAsync(ModelTypes type, string modelVariant, CancellationToken cancel = default)
    {
        var cacheKey = (type, modelVariant);
        if (CurrentModel is not null &&
            _variantCache.TryGetValue(cacheKey, out var ready) &&
            ready == CurrentModel)
            return;      // already active

        IsDownloading = true;
        DownloadProgress = 0;

        try
        {
            if (type == ModelTypes.FastVLM05B)
            {
                string dec = await EnsureFastVlmVariantAsync(modelVariant, new Progress<double>(p => DownloadProgress = p), cancel);
                
                // Get the file paths based on the variant
                var (embedPath, decoderPath) = GetFastVlmFilePaths(modelVariant);
                
                CurrentModel = await FastVLM.FastVlmSimple.CreateAsync(
                    Path.Combine(FileSystem.Current.AppDataDirectory, "vision_encoder.onnx"),
                    embedPath,
                    decoderPath,
                    Path.Combine(FileSystem.Current.AppDataDirectory, "vocab.json"),
                    Path.Combine(FileSystem.Current.AppDataDirectory, "merges.txt"),
                    cancel);
                _variantCache[cacheKey] = CurrentModel;
            }
            else
            {
                string localPath = await EnsureModelFileAsync(type, cancel);
                CurrentModel = CreateVisionWrapper(type, localPath);
                _variantCache[cacheKey] = CurrentModel;
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
    

    static async Task<string> EnsureFastVlmTypedAsync(IProgress<double> progress, CancellationToken ct, string type = "q4f16")
    {
        // Download different model variants based on type parameter
        // type: "" = full precision, "q4f16" = mixed, "f16" = all fp16, "int8" = all int8
        
        string root = "https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX/resolve/main/";
        
        // Determine file suffixes based on type
        string visionSuffix, embedSuffix, decoderSuffix;
        switch (type)
        {
            case "": // Full precision
                visionSuffix = "";
                embedSuffix = "";
                decoderSuffix = "";
                break;
            case "q4f16": // Mixed precision (current default)
                visionSuffix = "q4f16";
                embedSuffix = "q4f16";
                decoderSuffix = "q4f16";
                break;
            case "f16": // All FP16
                visionSuffix = "_fp16";
                embedSuffix = "_fp16";
                decoderSuffix = "_fp16";
                break;
            case "int8": // All INT8
                visionSuffix = "int8"; // Vision stays fp16 even in int8 mode
                embedSuffix = "_int8";
                decoderSuffix = "_int8";
                break;
            default:
                throw new ArgumentException($"Unknown FastVLM type: {type}");
        }

        // Build file names
        string visionFile = $"vision_encoder{visionSuffix}.onnx";
        string embedFile = $"embed_tokens{embedSuffix}.onnx";
        string decoderFile = $"decoder_model_merged{decoderSuffix}.onnx";
        
        // Download 6 files total, so each gets 1/6 of progress
        const int totalFiles = 6;
        double fileProgress = 1.0 / totalFiles;
        
        // Download all files with proper progress reporting
        _ = await ModelCache.GetAsync(root + "onnx/" + visionFile, visionFile, 
            new Progress<double>(p => progress.Report(0 * fileProgress + p * fileProgress)), ct);
            
        _ = await ModelCache.GetAsync(root + "onnx/" + embedFile, embedFile, 
            new Progress<double>(p => progress.Report(1 * fileProgress + p * fileProgress)), ct);
            
        string dec = await ModelCache.GetAsync(root + "onnx/" + decoderFile, decoderFile, 
            new Progress<double>(p => progress.Report(2 * fileProgress + p * fileProgress)), ct);
            
        _ = await ModelCache.GetAsync(root + "merges.txt", "merges.txt", 
            new Progress<double>(p => progress.Report(3 * fileProgress + p * fileProgress)), ct);
            
        _ = await ModelCache.GetAsync(root + "vocab.json", "vocab.json", 
            new Progress<double>(p => progress.Report(4 * fileProgress + p * fileProgress)), ct);
            
        _ = await ModelCache.GetAsync(root + "tokenizer.json", "tokenizer.json", 
            new Progress<double>(p => progress.Report(5 * fileProgress + p * fileProgress)), ct);
            
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
    readonly ConcurrentDictionary<(ModelTypes, string), IVision> _variantCache = new();

    static async Task<string> EnsureFastVlmVariantAsync(string variant, IProgress<double> progress, CancellationToken ct)
    {
        return await EnsureFastVlmTypedAsync(progress, ct, variant);
    }

    static (string embedPath, string decoderPath) GetFastVlmFilePaths(string variant)
    {
        // Determine file suffixes based on variant
        string embedSuffix, decoderSuffix;
        switch (variant)
        {
            case "": // Full precision
                embedSuffix = "";
                decoderSuffix = "";
                break;
            case "mixed":
            case "q4f16": // Mixed precision (current default)
                embedSuffix = "_int8";
                decoderSuffix = "_int8";
                break;
            case "f16": // All FP16
                embedSuffix = "_fp16";
                decoderSuffix = "_fp16";
                break;
            case "int8": // All INT8
                embedSuffix = "_int8";
                decoderSuffix = "_int8";
                break;
            default:
                throw new ArgumentException($"Unknown FastVLM variant: {variant}");
        }

        string embedFile = $"embed_tokens{embedSuffix}.onnx";
        string decoderFile = $"decoder_model_merged{decoderSuffix}.onnx";
        
        return (Path.Combine(FileSystem.Current.AppDataDirectory, embedFile),
                Path.Combine(FileSystem.Current.AppDataDirectory, decoderFile));
    }

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
    /// Helper method to quickly test different FastVLM model variants
    /// </summary>
    public async Task TestFastVlmVariantAsync(string variant, byte[] testImage, string prompt = "What do you see in this image?", CancellationToken cancel = default)
    {
        await ChangeModelAsync(ModelTypes.FastVLM05B, variant, cancel);
        string result = await AnalyzeWithVlmAsync(testImage, prompt, 20, cancel);
        System.Diagnostics.Debug.WriteLine($"FastVLM variant '{variant}' result: {result}");
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
