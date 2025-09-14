using Camera.MAUI;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Identity.Client;
using System.Diagnostics;

namespace PoopDetector.ViewModel;

public partial class LoginViewModel : ObservableObject
{
    public string Id => "user";

    public bool SignedIn
    {
        get => true;
    }

    [RelayCommand]
    async Task NavigateToRegisterPage()
    {
        await Shell.Current.GoToAsync(nameof(RegisterPage));
    }

    [RelayCommand]
    async Task Login()
    {
        try
        {

            OnPropertyChanged(nameof(SignedIn));
            // Handle successful authentication here
        }
        catch (MsalException ex)
        {
            Debug.WriteLine(ex.Message);
            // Handle authentication errors here
        }
        catch (Exception ex)
        {
            Debug.WriteLine(ex.Message);
            // Handle authentication errors here
        }
    }
}
