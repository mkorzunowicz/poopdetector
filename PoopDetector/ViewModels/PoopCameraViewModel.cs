using Camera.MAUI;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PoopDetector.AI;
using SignInMaui.MSALClient;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using PoopDetector.Models;
using System.Drawing;
using PoopDetector.AI.Vision;

namespace PoopDetector.ViewModel
{
    public partial class PoopCameraViewModel : ObservableObject
    {
        // existing fields
        private int _fps;
        private CameraInfo _selectedCamera;
        private ObservableCollection<CameraInfo> _cameras = new();

        // private readonly PoopPictureService _poopPictureService;
        private readonly CameraView cameraView;
        [ObservableProperty]
        private bool samResultReady;

        [ObservableProperty]
        private bool samRunning;
        public bool HasTorch => _selectedCamera != null && _selectedCamera.HasFlashUnit;

        [ObservableProperty]
        private bool torchOn;

        public PoopCameraViewModel(CameraView cameraView)
        {
            this.cameraView = cameraView;
           _modelTypes =  new ObservableCollection<VisionModelManager.ModelTypes>(Enum.GetValues<VisionModelManager.ModelTypes>());
        }


        // Indicates user can open settings only if more than one camera is available
        public bool ShowSelectModel => ModelTypes.Count > 1 && !FrozenPictureIsVisible;
        public bool ShowSettings => Cameras.Count > 1 && !FrozenPictureIsVisible;

        // If no cameras are available, show a “No cameras” UI
        public bool NoCamerasAvailable => Cameras.Count == 0;

        public bool PausePredictions { get; set; } = false;
        //[ObservableProperty]
        //public bool frozenPictureIsVisible;
        public bool FrozenPictureIsVisible { get; set; } = false;
        public bool CameraIsVisible { get; set; } = true;

        public bool CloserIsVisible
        {
            get
            {
                if (CurrentPrediction == null || CurrentPrediction.BoundingBoxes.Count == 0)
                    return false;

                return CurrentPrediction.IsTooSmall;
            }
        }
        public bool AwayIsVisible
        {
            get
            {
                if (CurrentPrediction == null || CurrentPrediction.BoundingBoxes.Count == 0)
                    return false;

                return CurrentPrediction.IsTooBig;
            }
        }
        public bool IsOnline()
        {
            var current = Connectivity.NetworkAccess;
            return current == NetworkAccess.Internet;
        }

        private async void Freeze()
        {
            if (VisionModelManager.Instance.MobileSam == null) return;
            PausePredictions = true;

            FrozenPictureIsVisible = true;
            CameraIsVisible = false;
            OnPropertyChanged(nameof(FrozenPictureImageSource));
            OnPropertyChanged(nameof(CameraIsVisible));
            OnPropertyChanged(nameof(FrozenPictureIsVisible));
            OnPropertyChanged(nameof(ShowSettings));

            //FrozenPicture = await MainThread.InvokeOnMainThreadAsync(async () =>
            //{
            //    //return await _poopPictureService.PreparePicture(CurrentPrediction);
            //});
            SamRunning = true;
            await CurrentPrediction.RunSamEncode();
            await CurrentPrediction.RunSamDecode();

            SamResultReady = true;
            SamRunning = false;
        }

        [RelayCommand]
        async Task ForcePicture()
        {
            await Task.Run(Freeze);
        }

        [RelayCommand]
        async Task ClearSamPoints()
        {
            await Task.Run(CurrentPrediction.ClearSamPoints);
            SamResultReady = true;
        }
        [RelayCommand]
        async Task CameraFocus()
        {
            cameraView.ForceAutoFocus();
        }

        [RelayCommand]
        async Task CancelPicture()
        {
            PausePredictions = false;
            FrozenPictureIsVisible = false;
            CameraIsVisible = true;
            OnPropertyChanged(nameof(CameraIsVisible));
            OnPropertyChanged(nameof(FrozenPictureIsVisible));
            OnPropertyChanged(nameof(ShowSettings));

        }

        public ImageSource FrozenPictureImageSource
        {
            get
            {
                if (CurrentPrediction == null || CurrentPrediction.InputImage == null)
                    return null;

                MemoryStream memory = new(CurrentPrediction.InputImage);
                return ImageSource.FromStream(() => memory);
            }
        }

        public int FPS
        {
            get => _fps;
            set
            {
                _fps = value;
                OnPropertyChanged();
            }
        }

        PoopPicture FrozenPicture { get; set; }

        [RelayCommand]
        async Task AcceptPicture()
        {
            try
            {
                if (IsOnline())
                {
                    //await _poopPictureService.SendPicture(FrozenPicture);
                }
                else
                {
                    // TODO cache it and send when internet is back!
                    Debug.WriteLine("No internet connection. Picture not saved or cached!");
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                // Handle authentication errors here
            }
        }

        public void AddPredictionResult(PredictionResult result)
        {
            int count = 5;
#if IOS
count = 15;
#endif
            if (LastPredictions.Count == count)
                LastPredictions.RemoveAt(0);
            LastPredictions.Add(result);
            CurrentPrediction = result;

            if (LastPredictions.Count == count && LastPredictions.All(p => p.IsGood))
            {
                // Freeze();
                LastPredictions.Clear();
            }
            else
            {
                OnPropertyChanged(nameof(AwayIsVisible));
                OnPropertyChanged(nameof(CloserIsVisible));
            }
        }

        public PredictionResult CurrentPrediction { get; internal set; }
        public List<PredictionResult> LastPredictions { get; internal set; } = new List<PredictionResult>(5);

        //public event PropertyChangedEventHandler PropertyChanged;

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
                if (_selectedCamera != value)
                {
                    SetProperty(ref _selectedCamera, value);
                    OnPropertyChanged(nameof(NoCamerasAvailable));
                    OnPropertyChanged(nameof(ShowSettings));
                    OnPropertyChanged(nameof(HasTorch));
                    SelectedCameraChanged?.Invoke(_selectedCamera);
                }
            }
        }

        private async void ChangeModelAsync(VisionModelManager.ModelTypes type)
        {
            await VisionModelManager.Instance.ChangeModelAsync(type);
        }
        private ObservableCollection<VisionModelManager.ModelTypes> _modelTypes ;
        public ObservableCollection<VisionModelManager.ModelTypes> ModelTypes => _modelTypes;

        // [RelayCommand]
        // async Task SelectModel()
        // {
        //     await Navigation.PushModalAsync(new CameraSelectionPage(_viewModel));
        // }
        private VisionModelManager.ModelTypes _selectedModelType;
        public VisionModelManager.ModelTypes SelectedModelType
        {
            get => _selectedModelType;
            set
            {
                if (_selectedModelType != value)
                {
                    _selectedModelType = value;
                    OnPropertyChanged();
                    ChangeModelAsync(value);
                }
            }
        }

    }
}
