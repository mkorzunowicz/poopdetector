using PoopDetector.ViewModel;

namespace PoopDetector.Views
{
    public partial class ModelSelectionPage : ContentPage
    {
        public ModelSelectionPage()
        {
            InitializeComponent();
            BindingContext = new ModelSelectionViewModel(this); 
        }
    }
}
