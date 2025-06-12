using Microsoft.ML.OnnxRuntime;
using PoopDetector.AI.Vision.Processing;
using PoopDetector.AI.Vision;
using Microsoft.Maui.Graphics;

namespace PoopDetector.AI;

public class DoubleVisionBase<TImageProcessor> : IVision where TImageProcessor : new()
{
    byte[] _model;
    byte[] _model2;
    string _name;
    string _name2;
    string _modelName;
    string _model2Name;
    Task _prevAsyncTask;
    TImageProcessor _imageProcessor;
    InferenceSession _session;
    InferenceSession _session2;

    public DoubleVisionBase(string name, string modelName, string name2, string model2Name)
    {
        _name = name;
        _modelName = modelName;
        _name2 = name2;
        _model2Name = model2Name;
        _ = InitializeAsync();
    }
    public virtual Size InputSize =>
        throw new NotImplementedException();
    public string Name => _name;
    public string Name2 => _name2;
    public string ModelName => _modelName;
    public string Model2Name => _model2Name;
    public byte[] Model => _model;
    public InferenceSession Session => _session;
    public InferenceSession Session2 => _session2;
    public TImageProcessor ImageProcessor => _imageProcessor ??= new TImageProcessor();

    public async Task UpdateExecutionProviderAsync(ExecutionProviders executionProvider)
    {
        // make sure any existing async task completes before we change the session
        await AwaitLastTaskAsync();

        // creating the inference session can be expensive and should be done as a one-off.
        // additionally each session uses memory for the model and the infrastructure required to execute it,
        // and has its own threadpools.
        _prevAsyncTask = Task.Run(() => NewSession(executionProvider));
    }

    private void NewSession(ExecutionProviders executionProvider)
    {
        var options = new SessionOptions();
        if (executionProvider == ExecutionProviders.CPU)
        {
            // CPU Execution Provider is always enabled
        }
        else if (executionProvider == ExecutionProviders.NNAPI)
        {
            options.AppendExecutionProvider_Nnapi();
        }
        else if (executionProvider == ExecutionProviders.OpenVINO)
        {
            options.AppendExecutionProvider_OpenVINO();
        }
        else if (executionProvider == ExecutionProviders.CoreML)
        {
            // add CoreML if the device has an Apple Neural Engine. if it doesn't performance
            // will most likely be worse than with the CPU Execution Provider.
            options.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE);
        }
        else if (executionProvider == ExecutionProviders.CUDA)
        {
            options.AppendExecutionProvider_CUDA(new OrtCUDAProviderOptions());
        }

        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        _session = new InferenceSession(_model, options);
        _session2 = new InferenceSession(_model2, options);
    }

    protected virtual Task<ImageProcessingResult> OnProcessImageAsync(byte[] image) =>
        throw new NotImplementedException();

    public Task InitializeAsync()
    {
        _prevAsyncTask = Initialize();
        return _prevAsyncTask;
    }
    public async Task<ImageProcessingResult> ProcessImageAsync(byte[] image)
    {
        await AwaitLastTaskAsync().ConfigureAwait(false);

        return await OnProcessImageAsync(image);
    }

    async Task AwaitLastTaskAsync()
    {
        if (_prevAsyncTask != null)
        {
            await _prevAsyncTask.ConfigureAwait(false);
            _prevAsyncTask = null;
        }
    }

    async Task Initialize()
    {
        _model = await Utils.LoadResource(_modelName);
        _model2 = await Utils.LoadResource(_model2Name);

        // This should allow use of NNAPI on android, but this ends up running slower than expected (~3 slower than CPU).
        // The model might need to be rebuilt with different configuration.
        // NNAPI is also deprecated
        if (DeviceInfo.Platform == DevicePlatform.iOS)
            NewSession(ExecutionProviders.CPU);
        //else if (DeviceInfo.Platform == DevicePlatform.Android)
        //    NewSession(ExecutionProviders.NNAPI);
        else
            NewSession(ExecutionProviders.CPU);
    }
}
