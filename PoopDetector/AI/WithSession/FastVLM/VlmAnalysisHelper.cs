using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Camera.MAUI;
using System.Diagnostics;

namespace PoopDetector.AI.Vision.FastVLM;

/// <summary>
/// Helper class to manage VLM analysis UI state and commands.
/// Extracted from PoopCameraViewModel to keep VLM-related code close to VLM implementation.
/// </summary>
public partial class VlmAnalysisHelper : ObservableObject
{
    private readonly CameraView _cameraView;
    private readonly Func<FastVlm?> _getVlmModel;

    public VlmAnalysisHelper(CameraView cameraView, Func<FastVlm?> getVlmModel)
    {
        _cameraView = cameraView;
        _getVlmModel = getVlmModel;
    }

    // --------- FastVLM prompt + output ---------
    [ObservableProperty]
    private string promptText = "Describe what you see";

    [ObservableProperty]
    private string lastVlmJson = string.Empty;

    // --------- Progressive Response Display ---------
    [ObservableProperty]
    private string progressiveResponse = string.Empty;

    [ObservableProperty]
    private bool isGeneratingResponse = false;

    // Property to control when answer box should be visible
    public bool ShouldShowAnswerBox => IsGeneratingResponse && !string.IsNullOrWhiteSpace(ProgressiveResponse);

    // Override property change to notify ShouldShowAnswerBox changes
    partial void OnProgressiveResponseChanged(string value)
    {
        OnPropertyChanged(nameof(ShouldShowAnswerBox));
    }

    partial void OnIsGeneratingResponseChanged(bool value)
    {
        OnPropertyChanged(nameof(ShouldShowAnswerBox));
    }

    [RelayCommand]
    private void ClearVlmResponse()
    {
        IsGeneratingResponse = false;
        ProgressiveResponse = string.Empty;
    }

    [RelayCommand]
    private async Task AnalyzeWithVlm()
    {
        try
        {
            var vlm = _getVlmModel();
            if (vlm == null)
            {
                ProgressiveResponse = "VLM model not loaded";
                return;
            }

            // Show progressive response UI and reset previous content
            IsGeneratingResponse = true;
            ProgressiveResponse = "?? Analyzing image...";

            // Capture single frame
            Stream? stream;
            if (DeviceInfo.Platform == DevicePlatform.Android || DeviceInfo.Platform == DevicePlatform.iOS)
                stream = _cameraView.GetSnapShotStream(Camera.MAUI.ImageFormat.JPEG);
            else
                stream = await _cameraView.TakePhotoAsync(Camera.MAUI.ImageFormat.JPEG);

            if (stream == null)
            {
                IsGeneratingResponse = false;
                ProgressiveResponse = "Failed to capture image";
                return;
            }

            using var ms = new MemoryStream();
            await stream.CopyToAsync(ms);

            // Use streaming API with callback to update progressive response
            await vlm.GenerateStreamAsync(
                ms.ToArray(),
                PromptText,
                400,
                (partialResponse) =>
                {
                    // Update UI on main thread
                    MainThread.BeginInvokeOnMainThread(() =>
                    {
                        ProgressiveResponse = partialResponse;
                    });
                }
            );

            // Final result
            LastVlmJson = ProgressiveResponse;

            // Keep the progressive response visible - user can manually dismiss or start new analysis
        }
        catch (Exception ex)
        {
            LastVlmJson = $"{DateTime.Now:o} error: {ex.Message}";
            ProgressiveResponse = $"Error: {ex.Message}";
            Debug.WriteLine($"[VlmAnalysisHelper] Error in AnalyzeWithVlm: {ex}");
        }
    }
}
