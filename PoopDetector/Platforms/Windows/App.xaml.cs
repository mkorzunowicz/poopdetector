using Microsoft.Identity.Client;
using Microsoft.UI.Xaml;
using SignInMaui.MSALClient;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace PoopDetector.WinUI
{
    /// <summary>
    /// Provides application-specific behavior to supplement the default Application class.
    /// </summary>
    public partial class App : MauiWinUIApplication
    {
        /// <summary>
        /// Initializes the singleton application object.  This is the first line of authored code
        /// executed, and as such is the logical equivalent of main() or WinMain().
        /// </summary>
        public App()
        {
            this.InitializeComponent();
            // configure redirect URI for your application
            //PlatformConfig.Instance.RedirectUri = $"msal{PublicClientSingleton.Instance.MSALClientHelper.AzureAdB2CConfig.ClientId}://auth";

            // Initialize MSAL
            IAccount existinguser = Task.Run(PublicClientSingleton.Instance.MSALClientHelper.InitializePublicClientAppAsync).Result;
            //if (existinguser != null)
            //    Task.Run(PublicClientSingleton.Instance.AcquireTokenSilentAsync);
        }

        protected override MauiApp CreateMauiApp() => MauiProgram.CreateMauiApp();

        protected override void OnLaunched(LaunchActivatedEventArgs args)
        {
            base.OnLaunched(args);

            var app = PoopDetector.App.Current;
            PlatformConfig.Instance.ParentWindow = ((MauiWinUIWindow)app.Windows[0].Handler.PlatformView).WindowHandle;
        }
    }
}
