// PoopDetector.AI.Vision/VisionModelManager.cs
// --------------------------------------------------------------
using CommunityToolkit.Mvvm.ComponentModel;
using PoopDetector.Services;
using PoopDetector.AI.Vision.YoloX;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace PoopDetector.AI.Vision;

public partial class VisionModelManager : ObservableObject
{
    static readonly VisionModelOptions _options = VisionModelOptionsLoader.Load();
    static readonly IReadOnlyDictionary<ModelTypes, string> _modelFileNames =
        new Dictionary<ModelTypes, string>
        {
            { ModelTypes.YoloxNanoPoop, "yolox_nano_poop_cropped_only_best.onnx" },
            { ModelTypes.Yolov9ScatSpotter, "yolov9_poop.onnx" },
            { ModelTypes.YoloxNano, "yolox_nano.onnx" },
            { ModelTypes.ShitspotterCustomV2, "shitspotter_custom_v2_epoch126.onnx" },
            { ModelTypes.ShitspotterCustomV5, "shitspotter-custom-v5-epoch_115.onnx" },
        };

    static readonly IReadOnlyDictionary<ModelTypes, string> _legacyModelUrls =
        new Dictionary<ModelTypes, string>
        {
            {
                ModelTypes.YoloxNanoPoop,
                "https://github.com/mkorzunowicz/poop_models/raw/refs/heads/main/yolox_nano_poop_cropped_only_best.onnx"
            },
            {
                ModelTypes.Yolov9ScatSpotter,
                "https://huggingface.co/erotemic/shitspotter-models/resolve/main/models/yolo-v9/shitspotter-simple-v3-run-v06-epoch%3D0032-step%3D000132-trainlosstrain_loss%3D7.603.onnx"
            },
            {
                ModelTypes.YoloxNano,
                "https://huggingface.co/yourbucket/yolox_nano.onnx"
            },
            {
                ModelTypes.ShitspotterCustomV2,
                "https://github.com/Erotemic/poop_models/raw/refs/heads/main/shitspotter_custom_v2_epoch126.onnx"
            },
            {
                ModelTypes.ShitspotterCustomV5,
                "https://raw.githubusercontent.com/Erotemic/poop_models/main/shitspotter-custom-v5-epoch_115.onnx"
            }
        };

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

        await EnsureBundledModelsAsync(cancel);

        IsDownloading = true;
        DownloadProgress = 0;

        try
        {
            string localPath = await EnsureModelFileAsync(type, cancel);
            CurrentModel = CreateVisionWrapper(type, localPath);
            _cache[type] = CurrentModel;
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
    bool _bootstrapped;
    bool _bundledPrepared;

    public async Task EnsureDefaultModelAsync()
    {
        MobileSam = new MobileSam.MobileSam();
        await EnsureBundledModelsAsync(CancellationToken.None);
        if (_bootstrapped || CurrentModel is not null) return;

        string name = _modelFileNames[ModelTypes.YoloxNanoPoop];
        string defaultUrl = GetRemoteUrl(ModelTypes.YoloxNanoPoop);
        string localPath = Path.Combine(FileSystem.Current.AppDataDirectory, name);

        if (!File.Exists(localPath))    // first app launch
        {
            IsDownloading = true;
            DownloadProgress = 0;
            try
            {
                localPath = await ModelCache.GetAsync(
                    defaultUrl,
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

    async Task EnsureBundledModelsAsync(CancellationToken cancel)
    {
        if (_bundledPrepared)
            return;

        _bundledPrepared = true;

        if (_options.BundledModels == null || _options.BundledModels.Count == 0)
            return;

        foreach (string fileName in _options.BundledModels
                     .Where(n => !string.IsNullOrWhiteSpace(n))
                     .Select(n => n.Trim())
                     .Distinct(StringComparer.OrdinalIgnoreCase))
        {
            try
            {
                await ModelCache.EnsurePackagedCopyAsync(fileName, cancel);
            }
            catch (Exception ex)
            {
                RaiseError($"Could not stage bundled model '{fileName}': {ex.Message}");
            }
        }
    }

    static async Task<string> EnsureModelFileAsync(ModelTypes t,
                                                   CancellationToken ct)
    {
        var p = new Progress<double>(d =>
            Instance.DownloadProgress = d);     // pushes into binding

        string url = GetRemoteUrl(t);
        string fileName = _modelFileNames[t];
        return await ModelCache.GetAsync(url, fileName, p, ct);
    }

    static string GetRemoteUrl(ModelTypes type)
    {
        if (!_modelFileNames.TryGetValue(type, out var fileName))
            throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown model type.");

        _legacyModelUrls.TryGetValue(type, out var fallback);
        string? resolved = _options.ResolveRemoteUrl(type.ToString(), fileName, fallback);

        if (string.IsNullOrWhiteSpace(resolved))
            throw new InvalidOperationException($"No remote URL configured for model '{type}'.");

        return resolved;
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

            ModelTypes.ShitspotterCustomV2 =>
                new YoloX.YoloX(modelPath, 416, 416, YoloXColormap.PoopList),

            ModelTypes.ShitspotterCustomV5 =>
                new YoloX.YoloX(modelPath, 416, 416, YoloXColormap.PoopList),

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
        ShitspotterCustomV2,
        ShitspotterCustomV5,
    }
}
