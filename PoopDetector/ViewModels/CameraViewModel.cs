using Camera.MAUI;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PoopDetector.AI;
using PoopDetector.Models;
using SignInMaui.MSALClient;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;

namespace PoopDetector.ViewModel;

public partial class CameraViewModel : ObservableObject
{
    private int _fps;
    private AIModelManager.ModelTypes _selectedModelType;
    private ObservableCollection<AIModelManager.ModelTypes> _modelTypes;
    private CameraInfo _selectedCamera;
    private ObservableCollection<CameraInfo> _cameras = [];

    public CameraViewModel()
    {
        _modelTypes = new ObservableCollection<AIModelManager.ModelTypes>(
         Enum.GetValues(typeof(AIModelManager.ModelTypes)) as AIModelManager.ModelTypes[]);
    }
    public CameraInfo SelectedCamera
    {
        get => _selectedCamera;
        set
        {
            if (_selectedCamera != value)
            {
                _selectedCamera = value;
                OnPropertyChanged();
                SelectedCameraChanged?.Invoke(_selectedCamera);
            }
        }
    }

    [RelayCommand]
    async Task SavePicture()
    {
        try
        {
            var location = await Geolocation.GetLastKnownLocationAsync();

            if (location == null)
            {
                // If no location found, get the current location with higher accuracy
                location = await Geolocation.GetLocationAsync(new GeolocationRequest(GeolocationAccuracy.High, TimeSpan.FromSeconds(10)));
            }

            if (location == null)
            {
                Debug.WriteLine("No GPS location available.");
                return;
            }

            // Prepare picture data
            var picture = new PoopPicture()
            {
                File = CurrentPrediction.InputImage,
                Status = "Pending",
                //UserId = PublicClientSingleton.Instance.MSALClientHelper.AuthResult.UniqueId,
                DateTime = DateTime.Now,
                Geolocation = $"{location.Latitude};{location.Longitude}",
                SubmissionType = SubmissionType.BeforeCleanup,
                BoundingBoxes = CurrentPrediction.BoundingBoxesToJson()
                //Confidence = CurrentPrediction.BoundingBoxes.
            };

            // TODO: Save the picture somewhere!
            Debug.WriteLine("Picture would be saved!");
        }
        catch (Exception ex)
        {
            Debug.WriteLine(ex.Message);
            // Handle authentication errors here
        }
    }
    public event Action<CameraInfo> SelectedCameraChanged;
    public int FPS
    {
        get => _fps;
        set
        {
            _fps = value;
            OnPropertyChanged();
        }
    }
    public ObservableCollection<CameraInfo> Cameras { get { return _cameras; } set { _cameras = value; } }
    public ObservableCollection<AIModelManager.ModelTypes> ModelTypes => _modelTypes;

    public AIModelManager.ModelTypes SelectedModelType
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

    //public byte[] CurrentPicture { get; internal set; }
    public PredictionResult CurrentPrediction { get; internal set; }

    private async void ChangeModelAsync(AIModelManager.ModelTypes type)
    {
        await AIModelManager.Instance.ChangeModelAsync(type);
    }
    public event PropertyChangedEventHandler PropertyChanged;

    //protected void OnPropertyChanged([CallerMemberName] string propertyName = null)
    //{
    //    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    //}
}
