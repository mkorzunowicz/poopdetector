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
        await ChangeModelAsync(type, "int8", CancellationToken.None);
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
                
                // Get all file paths based on the variant
                var (visionEncoderPath, embedPath, decoderPath) = GetFastVlmFilePaths(modelVariant);
                
                CurrentModel = await FastVLM.FastVlm.CreateAsync(
                    visionEncoderPath,
                    embedPath,
                    decoderPath,
                    Path.Combine(FileSystem.Current.AppDataDirectory, "tokenizer.json"),
                    //Path.Combine(FileSystem.Current.AppDataDirectory, "vocab.json"),
                    //Path.Combine(FileSystem.Current.AppDataDirectory, "merges.txt"),
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
        // Supports: "", q4f16, fp16, int8, q4, bnb4, quantized
        
        string root = "https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX/resolve/main/";
        
        // Determine file suffixes based on type
        string visionSuffix, embedSuffix, decoderSuffix;
        switch (type)
        {
            case "": // Full precision (original)
                visionSuffix = "";
                embedSuffix = "";
                decoderSuffix = "";
                break;
            case "q4f16": // Mixed precision (current default)
                visionSuffix = "_q4f16";
                embedSuffix = "_q4f16";
                decoderSuffix = "_q4f16";
                break;
            case "fp16":
            case "f16": // All FP16
                visionSuffix = "_fp16";
                embedSuffix = "_fp16";
                decoderSuffix = "_fp16";
                break;
            case "int8": // All INT8
                visionSuffix = "_fp16"; // Vision encoder typically stays fp16
                embedSuffix = "_int8";
                decoderSuffix = "_int8";
                break;
            case "q4": // 4-bit quantized
                visionSuffix = "_q4";
                embedSuffix = "_q4";
                decoderSuffix = "_q4";
                break;
            case "bnb4": // BitsAndBytes 4-bit
            // smalles working combo
                visionSuffix = "_bnb4";
                // embedSuffix = "_bnb4";
                // decoderSuffix = "_bnb4";
                embedSuffix = "_int8";
                decoderSuffix = "_int8";
                break;
            case "quantized": // Generic quantized
            // seems fastest
                visionSuffix = "_quantized";
                embedSuffix = "_quantized";
                decoderSuffix = "_quantized";
                break;
            default:
                throw new ArgumentException($"Unknown FastVLM variant: {type}. Supported variants: '', 'q4f16', 'fp16', 'int8', 'q4', 'bnb4', 'quantized'");
        }

        // Build file names - keep variant-specific names for all files
        string visionFile = $"vision_encoder{visionSuffix}.onnx";
        string embedFile = $"embed_tokens{embedSuffix}.onnx";
        string decoderFile = $"decoder_model_merged{decoderSuffix}.onnx";
        
        // Download files with variant-specific names to avoid overwriting
        const int totalFiles = 6;
        double fileProgress = 1.0 / totalFiles;
        
        // Download vision encoder with variant-specific name
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

    static (string visionEncoderPath, string embedPath, string decoderPath) GetFastVlmFilePaths(string variant)
    {
        // Determine file suffixes based on variant
        // Supports: "", q4f16, fp16, int8, q4, bnb4, quantized
        string visionSuffix, embedSuffix, decoderSuffix;
        switch (variant)
        {
            case "": // Full precision (original)
                visionSuffix = "";
                embedSuffix = "";
                decoderSuffix = "";
                break;
            case "mixed":
            case "q4f16": // Mixed precision (current default)
                visionSuffix = "_q4f16";
                embedSuffix = "_q4f16";
                decoderSuffix = "_q4f16";
                break;
            case "fp16":
            case "f16": // All FP16
                visionSuffix = "_fp16";
                embedSuffix = "_fp16";
                decoderSuffix = "_fp16";
                break;
            case "int8": // All INT8
                visionSuffix = "_fp16"; // Vision encoder typically stays fp16
                embedSuffix = "_int8";
                decoderSuffix = "_int8";
                break;
            case "q4": // 4-bit quantized
                visionSuffix = "_q4";
                embedSuffix = "_q4";
                decoderSuffix = "_q4";
                break;
            case "bnb4": // BitsAndBytes 4-bit - using your working combo
                visionSuffix = "_bnb4";
                embedSuffix = "_int8";  // Use int8 embeddings (your fix)
                decoderSuffix = "_int8"; // Use int8 decoder (your fix)
                break;
            case "quantized": // Generic quantized
                visionSuffix = "_quantized";
                embedSuffix = "_quantized";
                decoderSuffix = "_quantized";
                break;
            default:
                throw new ArgumentException($"Unknown FastVLM variant: {variant}. Supported variants: '', 'mixed', 'q4f16', 'fp16', 'int8', 'q4', 'bnb4', 'quantized'");
        }

        string visionFile = $"vision_encoder{visionSuffix}.onnx";
        string embedFile = $"embed_tokens{embedSuffix}.onnx";
        string decoderFile = $"decoder_model_merged{decoderSuffix}.onnx";
        
        return (Path.Combine(FileSystem.Current.AppDataDirectory, visionFile),
                Path.Combine(FileSystem.Current.AppDataDirectory, embedFile),
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
        if (CurrentModel is FastVLM.FastVlm vlm)
            return await vlm.GenerateAsync(image, prompt, maxNewTokens, cancel);
        throw new InvalidOperationException("Current model is not a VLM");
    }

    // Streaming version that calls onTokenGenerated for each new token
    public async Task AnalyzeWithVlmStreamAsync(byte[] image, string prompt, int maxNewTokens, Action<string> onTokenGenerated, CancellationToken cancel = default)
    {
        if (CurrentModel is FastVLM.FastVlm vlm)
            await vlm.GenerateStreamAsync(image, prompt, maxNewTokens, onTokenGenerated, cancel);
        else
            throw new InvalidOperationException("Current model is not a streaming VLM");
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
    /// Get all available FastVLM model variants
    /// </summary>
    public static string[] GetAvailableFastVlmVariants()
    {
        return new string[] 
        { 
            "",        // Full precision (original)
            "q4f16",   // Mixed precision (default)
            "fp16",    // All FP16
            "int8",    // All INT8
            "q4",      // 4-bit quantized
            "bnb4",    // BitsAndBytes 4-bit
            "quantized" // Generic quantized
        };
    }

    /// <summary>
    /// Get human-readable descriptions for FastVLM variants
    /// </summary>
    public static Dictionary<string, string> GetFastVlmVariantDescriptions()
    {
        return new Dictionary<string, string>
        {
            [""] = "Full Precision (Original) - Highest quality, largest size",
            ["q4f16"] = "Q4F16 Mixed Precision - Good balance of speed/quality (Default)",
            ["fp16"] = "Half Precision (FP16) - Faster inference, moderate size",
            ["int8"] = "8-bit Integer - Fast inference, smaller size",
            ["q4"] = "4-bit Quantized - Very fast, compact size",
            ["bnb4"] = "BitsAndBytes 4-bit - Optimized 4-bit quantization",
            ["quantized"] = "Generic Quantized - General purpose quantization"
        };
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
