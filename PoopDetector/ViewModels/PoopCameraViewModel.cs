// File: ViewModel/PoopCameraViewModel.cs 
using Camera.MAUI;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Maui.Networking;
using PoopDetector.AI;
using PoopDetector.AI.Vision;
using PoopDetector.Models;
using PoopDetector.Services;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Color = System.Drawing.Color;
using PointF = Microsoft.Maui.Graphics.PointF;

namespace PoopDetector.ViewModel;

public partial class PoopCameraViewModel : ObservableObject
{
    // ────────────────────────── fields ──────────────────────────
    private readonly CameraView _cameraView;
    private readonly PoopPictureStorageService _storage = new();

    private double _fps;
    private CameraInfo _selectedCamera;
    private ObservableCollection<CameraInfo> _cameras = new();

    // ────────────────────────── ctor ────────────────────────────
    public PoopCameraViewModel(CameraView cameraView)
    {
        _cameraView = cameraView;
        _modelTypes = new ObservableCollection<VisionModelManager.ModelTypes>(Enum.GetValues<VisionModelManager.ModelTypes>());
        VisionModelManager.Instance.DownloadError +=
        async (_, msg) => await MainThread.InvokeOnMainThreadAsync(
            () => Application.Current.MainPage
                      .DisplayAlert("Download error", msg, "OK"));
    }

    // ────────────────────────── observable props ────────────────
    [ObservableProperty] private bool samResultReady;
    [ObservableProperty] private bool samRunning;
    [ObservableProperty] private bool torchOn;

    // hook to refresh AcceptPictureCommand
    //partial void OnSamResultReadyChanged(bool value) =>
    //    AcceptPictureCommand.NotifyCanExecuteChanged();

    // ────────────────────────── public state helpers ────────────
    public bool HasTorch => _selectedCamera?.HasFlashUnit == true;
    public bool PausePredictions { get; set; }
    public bool FrozenPictureIsVisible { get; set; }
    public bool CameraIsVisible { get; set; } = true;

    public bool ShowSelectModel => ModelTypes.Count > 1 && !FrozenPictureIsVisible;
    public bool ShowSettings => Cameras.Count > 1 && !FrozenPictureIsVisible;
    public bool NoCamerasAvailable => Cameras.Count == 0;

    public double FPS
    {
        get => _fps;
        set => SetProperty(ref _fps, Math.Round(value,2));
    }

    // ────────────────────────── properties for binding ──────────
    public ImageSource FrozenPictureImageSource =>
        CurrentPrediction?.InputImage is null ? null :
        ImageSource.FromStream(() => new MemoryStream(CurrentPrediction.InputImage));

    public ObservableCollection<CameraInfo> Cameras
    {
        get => _cameras;
        set
        {
            SetProperty(ref _cameras, value);
            OnPropertyChanged(nameof(NoCamerasAvailable));
            OnPropertyChanged(nameof(ShowSettings));
        }
    }

    public event Action<CameraInfo> SelectedCameraChanged;

    public CameraInfo SelectedCamera
    {
        get => _selectedCamera;
        set
        {
            if (SetProperty(ref _selectedCamera, value))
            {
                OnPropertyChanged(nameof(HasTorch));
                OnPropertyChanged(nameof(ShowSettings));
                SelectedCameraChanged?.Invoke(value);

                if (DeviceInfo.Platform != DevicePlatform.WinUI)
                    Application.Current.MainPage.Navigation.PopModalAsync();
            }
        }
    }

    private readonly ObservableCollection<VisionModelManager.ModelTypes> _modelTypes;
    public ObservableCollection<VisionModelManager.ModelTypes> ModelTypes => _modelTypes;

    private VisionModelManager.ModelTypes _selectedModelType;
    public VisionModelManager.ModelTypes SelectedModelType
    {
        get => _selectedModelType;
        set
        {
            if (SetProperty(ref _selectedModelType, value))
                _ = VisionModelManager.Instance.ChangeModelAsync(value);
        }
    }

    // ────────────────────────── prediction storage ──────────────
    public PredictionResult CurrentPrediction { get; private set; }
    public List<PredictionResult> LastPredictions { get; } = new(5);

    // ────────────────────────── UI helpers ──────────────────────
    public bool CloserIsVisible =>
        CurrentPrediction?.BoundingBoxes?.Count > 0 &&
        CurrentPrediction.IsTooSmall;

    public bool AwayIsVisible =>
        CurrentPrediction?.BoundingBoxes?.Count > 0 &&
        CurrentPrediction.IsTooBig;

    // ────────────────────────── commands ────────────────────────
    [RelayCommand] private Task ForcePicture() => Task.Run(Freeze);

    [RelayCommand]
    async Task ClearSamPoints()
    {
        await Task.Run(CurrentPrediction.ClearSamPoints);
        SamResultReady = true;
        OnPropertyChanged(nameof(CanAcceptPicture));
    }
    [RelayCommand] private Task CameraFocus() => Task.Run(_cameraView.ForceAutoFocus);

    [RelayCommand]
    private Task CancelPicture()
    {
        PausePredictions = false;
        FrozenPictureIsVisible = false;
        CameraIsVisible = true;
        OnPropertyChanged(nameof(CameraIsVisible));
        OnPropertyChanged(nameof(FrozenPictureIsVisible));
        OnPropertyChanged(nameof(ShowSettings));
        return Task.CompletedTask;
    }

    // Accept is disabled when no mask is present  ────────────────
    [RelayCommand]
    private async Task AcceptPicture()
    {
        try
        {
            var imgBytes = CurrentPrediction.InputImage;
            var maskBmp = CurrentPrediction.MaskBitmaps.First();

            await _storage.SaveAsync(imgBytes, maskBmp);

            // tell the gallery to refresh
            MessagingCenter.Send<object>(this, "PictureSaved");
        }
        catch (Exception ex)
        {
            Debug.WriteLine(ex);
        }
        finally
        {
            await CancelPicture();
        }
    }

    private bool CanAcceptPicture() =>
        CurrentPrediction?.MaskBitmaps?.Count > 0;

    // ────────────────────────── main logic ──────────────────────
    private async void Freeze()
    {
        if (VisionModelManager.Instance.MobileSam is null) return;

        PausePredictions = true;
        FrozenPictureIsVisible = true;
        CameraIsVisible = false;
        OnPropertyChanged(nameof(FrozenPictureImageSource));
        OnPropertyChanged(nameof(CameraIsVisible));
        OnPropertyChanged(nameof(FrozenPictureIsVisible));
        OnPropertyChanged(nameof(ShowSettings));

        SamRunning = true;
        await CurrentPrediction.RunSamEncode();
        await CurrentPrediction.RunSamDecode();
        SamRunning = false;
        OnPropertyChanged(nameof(CanAcceptPicture));
        SamResultReady = true;      // triggers UI & can-execute refresh
    }

    public void AddPredictionResult(PredictionResult result)
    {
        int max = 5;
#if IOS
        max = 15;
#endif
        if (LastPredictions.Count == max) LastPredictions.RemoveAt(0);
        LastPredictions.Add(result);
        CurrentPrediction = result;

        OnPropertyChanged(nameof(AwayIsVisible));
        OnPropertyChanged(nameof(CloserIsVisible));
    }

    public bool IsOnline() =>
        Connectivity.NetworkAccess == NetworkAccess.Internet;
}
