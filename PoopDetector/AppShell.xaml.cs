using Microsoft.Maui;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Controls.Xaml;
using PoopDetector.ViewModel;

namespace PoopDetector
{
    public partial class AppShell : Shell
    {
        AppShellViewModel _vm;
        public AppShell()
        {
            InitializeComponent();
            BindingContext = _vm = IPlatformApplication.Current.Services.GetService<AppShellViewModel>();
            Routing.RegisterRoute(nameof(LoginPage), typeof(LoginPage));
            Routing.RegisterRoute(nameof(RegisterPage), typeof(RegisterPage));
            Application.Current.UserAppTheme = Application.Current.UserAppTheme == AppTheme.Unspecified ? Application.Current.RequestedTheme : AppTheme.Light;
            ThemeToggleSwitch.IsEnabled = Application.Current.UserAppTheme == AppTheme.Dark;

        }

        private void OnThemeToggled(object sender, ToggledEventArgs e)
        {
            var isDarkMode = e.Value;
            Application.Current.UserAppTheme = isDarkMode ? AppTheme.Light : AppTheme.Dark;
        }
    }
}
