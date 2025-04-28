using Camera.MAUI;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Identity.Client;
using SignInMaui.MSALClient;
using System.Diagnostics;

namespace PoopDetector.ViewModel;

public partial class LoginViewModel : ObservableObject
{
    public string Id => PublicClientSingleton.Instance.MSALClientHelper.AuthResult.UniqueId;

    public bool SignedIn
    {
        get => PublicClientSingleton.Instance.MSALClientHelper.AuthResult != null;
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
            var result = await PublicClientSingleton.Instance.MSALClientHelper.SignInUserAndAcquireAccessToken(["openid", "profile", "email"]);

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
