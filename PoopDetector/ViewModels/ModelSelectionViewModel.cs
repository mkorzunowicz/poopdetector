// Compilation unit namespace must match your project’s namespaces.

using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PoopDetector.AI;
using PoopDetector.AI.Vision;
using System.Collections.ObjectModel;
using Backend = PoopDetector.AI.Vision.VisionModelManager.Backend;
using ModelTypes = PoopDetector.AI.Vision.VisionModelManager.ModelTypes;

namespace PoopDetector.ViewModel;
using CommunityToolkit.Mvvm.ComponentModel;

public partial class PresetOption : ObservableObject
{
    public ModelTypes Kind { get; }

    [ObservableProperty] string name;
    [ObservableProperty] bool isSelected;

    public PresetOption(ModelTypes kind)
    {
        Kind = kind;
        Name = kind.ToString();           // or a nicer user-friendly string
    }
}
/// <summary>
/// View-model for <see cref="Views.ModelSelectionPage"/>.
/// </summary>
public partial class ModelSelectionViewModel : ObservableObject
{
    // --------------- preset list (old behaviour) ------------------ //
    public ObservableCollection<PresetOption> Presets { get; }

    [ObservableProperty] PresetOption? selectedPreset;


    void OnPresetChanged(ModelTypes? value)
    {
        if (value == null) return;
        LoadPresetCommand.Execute(value);
    }

    // --------------- load any remote model ----------------------- //
    public ObservableCollection<Backend> BackendOptions { get; } =
        new(Enum.GetValues<Backend>());

    [ObservableProperty] string customUrl = string.Empty;
    [ObservableProperty] Backend selectedBackend = Backend.YoloX;
    [ObservableProperty] string inputWidth = "640";
    [ObservableProperty] string inputHeight = "640";

    // --------------- Commands ------------------------------------ //
    public IRelayCommand<PresetOption> LoadPresetCommand { get; }
    public IRelayCommand LoadRemoteCommand { get; }
    public IRelayCommand ClosePageCommand { get; }

    readonly Page _page;   // needed to close the modal

    public ModelSelectionViewModel(Page page)
    {
        _page = page;

        LoadPresetCommand = new AsyncRelayCommand<PresetOption>(LoadPresetAsync);
        LoadRemoteCommand = new AsyncRelayCommand(LoadRemoteAsync);
        ClosePageCommand = new AsyncRelayCommand(ClosePage);

        Presets = new ObservableCollection<PresetOption>(
            Enum.GetValues<ModelTypes>().Select(t => new PresetOption(t)));
        Presets[0].IsSelected = true;
        PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(SelectedPreset) && SelectedPreset != null)
                _ = LoadPresetAsync(SelectedPreset, CancellationToken.None);
        };
    }

    // ---------------- helpers ------------------------------------ //
    async Task LoadPresetAsync(PresetOption option, CancellationToken ct)
    {
        try
        {
            await VisionModelManager.Instance.ChangeModelAsync(option.Kind, ct);

            if (DeviceInfo.Platform != DevicePlatform.WinUI)
                await MainThread.InvokeOnMainThreadAsync(async () =>
                {
                    await ClosePage(ct);
                });
        }
        catch (Exception) { /* ignore */ } // ignore if already closed
        foreach (var p in Presets) p.IsSelected = p == option;
    }

    // ---------------- helpers ------------------------------------ //
    async Task ClosePage(CancellationToken ct)
    {
        // make sure this page wasn’t closed elsewhere
        if (Application.Current.MainPage?.Navigation.ModalStack
                        .Contains(_page) == true)
        {
            await Application.Current.MainPage.Navigation.PopModalAsync();
        }
    }
    async Task LoadRemoteAsync(CancellationToken ct)
    {
        if (!Uri.TryCreate(CustomUrl, UriKind.Absolute, out var uri))
        {
            await _page.DisplayAlert("Invalid URL", "Enter a valid http / https link.", "OK");
            return;
        }

        if (!int.TryParse(InputWidth, out int w) || w <= 0 ||
            !int.TryParse(InputHeight, out int h) || h <= 0)
        {
            await _page.DisplayAlert("Size error", "Input W/H must be positive integers.", "OK");
            return;
        }

        // choose a colour map – for demo we always use PoopList
        var labels = YoloXColormap.PoopList;

        await VisionModelManager.Instance.LoadRemoteModelAsync(
                uri.ToString(),
                SelectedBackend,
                w, h,
                labels,
                ct);

        await ClosePage(ct);
    }
}
