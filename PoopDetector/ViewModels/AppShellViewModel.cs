using Camera.MAUI;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Identity.Client;
using SignInMaui.MSALClient;
using System.Diagnostics;

namespace PoopDetector.ViewModel;

public partial class AppShellViewModel : ObservableObject
{
    public AppShellViewModel()
    {
    }
    public string Id => PublicClientSingleton.Instance.MSALClientHelper.AuthResult.UniqueId;

    public bool LoggedinAndRefreshed
    {
        get => PublicClientSingleton.Instance.MSALClientHelper.AuthResult != null;
    }
    public bool LoggedIn
    {
        get => PublicClientSingleton.Instance.MSALClientHelper.Account != null;
    }

    public Visibility LoginButtonVisibility
    {
        get => LoggedIn ? Visibility.Visible : Visibility.Collapsed;
    }

    [RelayCommand]
    async Task Login()
    {
        try
        {
            var result = await PublicClientSingleton.Instance.AcquireTokenSilentAsync();
            //var result = await PublicClientSingleton.Instance.MSALClientHelper.SignInUserAndAcquireAccessToken(PublicClientSingleton.Instance.GetScopes());
            OnPropertyChanged(nameof(LoginButtonVisibility));
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
