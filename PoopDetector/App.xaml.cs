#if WINDOWS
using Microsoft.UI;
using Microsoft.UI.Windowing;
using Windows.Graphics;
#endif
using Microsoft.Identity.Client;
using Microsoft.Maui.Controls;
using PoopDetector.AI.Vision;

namespace PoopDetector
{
    public partial class App : Application
    {
        const int WindowWidth = 700;
        const int WindowHeight = 1000;

        public static IPublicClientApplication PublicClientApp { get; private set; }
        public App()
        {
            InitializeComponent();

            Microsoft.Maui.Handlers.WindowHandler.Mapper.AppendToMapping(nameof(IWindow), (handler, view) =>
            {
#if WINDOWS
            var mauiWindow = handler.VirtualView;
            var nativeWindow = handler.PlatformView;
            nativeWindow.Activate();
            IntPtr windowHandle = WinRT.Interop.WindowNative.GetWindowHandle(nativeWindow);
            WindowId windowId = Microsoft.UI.Win32Interop.GetWindowIdFromWindow(windowHandle);
            AppWindow appWindow = Microsoft.UI.Windowing.AppWindow.GetFromWindowId(windowId);
            appWindow.Resize(new SizeInt32(WindowWidth, WindowHeight));
#endif
            });
          
            MainPage = new AppShell();
        }
        protected override async void OnStart()
        {
            base.OnStart();

            await VisionModelManager.Instance.EnsureDefaultModelAsync();
        }
    }
}
