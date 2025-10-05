using Foundation;

using Microsoft.Identity.Client;
using UIKit;

namespace PoopDetector
{
    [Register("AppDelegate")]
    public class AppDelegate : MauiUIApplicationDelegate
    {
        protected override MauiApp CreateMauiApp() => MauiProgram.CreateMauiApp();

        public override bool FinishedLaunching(UIApplication application, NSDictionary launchOptions)
        {
            // configure platform specific params

            // Initialize MSAL and platformConfig is set

            return base.FinishedLaunching(application, launchOptions);
        }
    }
}
