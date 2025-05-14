using PoopDetector.Models;
using PoopDetector.ViewModel;

namespace PoopDetector.Views;

public partial class PictureGalleryPage : ContentPage
{
    private PictureGalleryViewModel Vm => (PictureGalleryViewModel)BindingContext;

    public PictureGalleryPage()
    {
        InitializeComponent();
        BindingContext = new PictureGalleryViewModel();
    }

    protected override void OnAppearing()
    {
        base.OnAppearing();
        _ = Vm.RefreshCommand.ExecuteAsync(null); // fallback auto-refresh
    }

    private async void OnSelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (e.CurrentSelection.FirstOrDefault() is SavedPoopPicture pic)
        {
            ((CollectionView)sender).SelectedItem = null;
            await Navigation.PushAsync(new PictureDetailPage(pic));
        }
    }
}
