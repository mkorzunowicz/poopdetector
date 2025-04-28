using CommunityToolkit.Mvvm.Input;

namespace PoopDetector.ViewModel;

public partial class RegisterViewModel
{
    [RelayCommand]
    async Task NavigateToRegisterPage()
    {
        await Shell.Current.GoToAsync(nameof(RegisterPage));
    }
}
