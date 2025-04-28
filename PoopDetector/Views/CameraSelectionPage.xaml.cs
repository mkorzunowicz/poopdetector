using PoopDetector.ViewModel;

namespace PoopDetector.Views
{
    public partial class CameraSelectionPage : ContentPage
    {
        private PoopCameraViewModel _viewModel;

        public CameraSelectionPage(PoopCameraViewModel vm)
        {
            InitializeComponent();
            _viewModel = vm;
            BindingContext = _viewModel;
        }

        private async void CloseButton_Clicked(object sender, EventArgs e)
        {
            await Navigation.PopModalAsync();
        }
    }
}
