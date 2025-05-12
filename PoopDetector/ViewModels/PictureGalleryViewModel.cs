// File: ViewModels/PictureGalleryViewModel.cs
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PoopDetector.Models;
using PoopDetector.Services;
using System.Collections.ObjectModel;
using System.Threading.Tasks;

namespace PoopDetector.ViewModel;

public partial class PictureGalleryViewModel : ObservableObject
{
    private readonly PoopPictureStorageService _storage = new();

    public ObservableCollection<SavedPoopPicture> Pictures { get; } = [];

    public PictureGalleryViewModel()
    {
        _ = LoadAsync();
    }

    private async Task LoadAsync()
    {
        var list = await _storage.GetAllAsync();
        MainThread.BeginInvokeOnMainThread(() =>
        {
            Pictures.Clear();
            foreach (var p in list) Pictures.Add(p);
        });
    }

    [RelayCommand]
    private async Task Refresh() => await LoadAsync();
}
