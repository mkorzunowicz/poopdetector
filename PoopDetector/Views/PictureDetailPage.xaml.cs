// File: Views/PictureDetailPage.xaml.cs
using PoopDetector.Models;

namespace PoopDetector.Views;

public partial class PictureDetailPage : ContentPage
{
    public PictureDetailPage(SavedPoopPicture pic)
    {
        InitializeComponent();
        Image.Source = pic.ImagePath;
        Mask.Source = pic.MaskPath;
    }
}
