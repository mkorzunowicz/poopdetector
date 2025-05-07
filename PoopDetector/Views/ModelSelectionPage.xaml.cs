using PoopDetector.ViewModel;

namespace PoopDetector.Views
{
    public partial class ModelSelectionPage : ContentPage
    {
        private PoopCameraViewModel _viewModel;

        public ModelSelectionPage(PoopCameraViewModel vm)
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
