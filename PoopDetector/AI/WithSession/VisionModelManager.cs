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
    public RepVitSam.RepVitSam? RepVitSam { get; private set; }
    public EdgeSam.EdgeSam? EdgeSam { get; private set; }

    // Active SAM model - easy switching between implementations
    private SamModelType _activeSamType = SamModelType.EdgeSam;
    
    public enum SamModelType
    {
        MobileSam,
        RepVitSam,
        EdgeSam
    }
    
    /// <summary>
    /// Gets the currently active SAM model
    /// </summary>
    public ISamModel? ActiveSam => _activeSamType switch
    {
        SamModelType.MobileSam => MobileSam,
        SamModelType.RepVitSam => RepVitSam,
        SamModelType.EdgeSam => EdgeSam,
        _ => MobileSam
    };
    
    /// <summary>
    /// Switch between SAM implementations
    /// </summary>
    public void UseMobileSam() => _activeSamType = SamModelType.MobileSam;
    public void UseRepVitSam() => _activeSamType = SamModelType.RepVitSam;
    public void UseEdgeSam() => _activeSamType = SamModelType.EdgeSam;

    // --------------  progress binding properties  -------------- //
    [ObservableProperty] double _downloadProgress;  // 0-1
    [ObservableProperty] bool _isDownloading;
    [ObservableProperty] bool _isVlmModelLoaded;
    
    private ModelTypes? _currentModelType;

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
                // Use FastVlmModelManager for all FastVLM-specific logic
                _ = await FastVLM.FastVlmModelManager.DownloadVariantAsync(
                    modelVariant, 
                    new Progress<double>(p => DownloadProgress = p), 
                    cancel);
                
                var (visionEncoderPath, embedPath, decoderPath) = FastVLM.FastVlmModelManager.GetFilePaths(modelVariant);
                
                CurrentModel = await FastVLM.FastVlm.CreateAsync(
                    visionEncoderPath,
                    embedPath,
                    decoderPath,
                    Path.Combine(FileSystem.Current.AppDataDirectory, "tokenizer.json"),
                    cancel);
                _variantCache[cacheKey] = CurrentModel;
                _currentModelType = type;
                IsVlmModelLoaded = true;
            }
            else
            {
                string localPath = await EnsureModelFileAsync(type, cancel);
                CurrentModel = CreateVisionWrapper(type, localPath);
                _variantCache[cacheKey] = CurrentModel;
                _currentModelType = type;
                IsVlmModelLoaded = false;
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

    const string _defaultUrl =
    "https://github.com/mkorzunowicz/poop_models/raw/refs/heads/main/" +
    "yolox_nano_poop_cropped_only_best.onnx";

    bool _bootstrapped;

    public async Task EnsureDefaultModelAsync()
    {
        MobileSam = new MobileSam.MobileSam();
        RepVitSam = new RepVitSam.RepVitSam();
        EdgeSam = new EdgeSam.EdgeSam();
        
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

    static async Task<string> EnsureModelFileAsync(ModelTypes t, CancellationToken ct)
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
    /// <param name="inputW">Model's expected width  (default 640)</param>
    /// <param name="inputH">Model's expected height (default 640)</param>
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
            throw;
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
