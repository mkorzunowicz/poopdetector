// File: Views/PictureGalleryPage.xaml.cs
using PoopDetector.Models;
using PoopDetector.ViewModel;

namespace PoopDetector.Views;

public partial class PictureGalleryPage : ContentPage
{
    public PictureGalleryPage()
    {
        InitializeComponent();
        BindingContext = new PictureGalleryViewModel();
    }

    private async void OnSelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (e.CurrentSelection.FirstOrDefault() is SavedPoopPicture pic)
        {
            ((CollectionView)sender).SelectedItem = null;   // clear selection
            await Navigation.PushAsync(new PictureDetailPage(pic));
        }
    }
}
