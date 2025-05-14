// File: ViewModel/PictureGalleryViewModel.cs   (replaces old file)
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Maui.Controls;
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
        // initial load
        _ = LoadAsync();
        //CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.
        // listen for newly saved pictures
        MessagingCenter.Subscribe<object>(this, "PictureSaved", async _ => await LoadAsync());
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

    [RelayCommand] private Task Refresh() => LoadAsync();
}
