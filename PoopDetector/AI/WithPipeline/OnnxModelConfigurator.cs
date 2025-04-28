using Microsoft.ML;
using Microsoft.ML.Transforms.Image;
using System.Diagnostics;

namespace PoopDetector.AI;

public class OnnxModelConfigurator<T> where T : class, IImageInputData
{
    private readonly MLContext mlContext;
    private readonly ITransformer mlModel;

    public OnnxModelConfigurator(IOnnxModel onnxModel)
    {
        mlContext = new MLContext();
        // Model creation and pipeline definition for images needs to run just once,
        // so calling it from the constructor:
        mlModel = SetupMlNetModel<T>(onnxModel);
    }

    private ITransformer SetupMlNetModel<T>(IOnnxModel onnxModel) where T : class, IImageInputData
    {        
        var dataView = mlContext.Data.LoadFromEnumerable(new List<T>());
        try
        {
            var pipeline = mlContext.Transforms.ResizeImages(resizing: ImageResizingEstimator.ResizingKind.Fill, outputColumnName: onnxModel.ModelInput, imageWidth: onnxModel.InputWidth, imageHeight: onnxModel.InputHeight, inputColumnName: nameof(IImageInputData.Image))
                        .Append(mlContext.Transforms.ExtractPixels(outputColumnName: onnxModel.ModelInput))
                        //.Append(mlContext.Transforms.ApplyOnnxModel(modelFile: onnxModel.ModelPath, outputColumnName: onnxModel.ModelOutput, inputColumnName: onnxModel.ModelInput, gpuDeviceId: 0, fallbackToCpu: true));
                        .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: onnxModel.ModelPath, outputColumnName: onnxModel.ModelOutput, inputColumnName: onnxModel.ModelInput));

            //mlContext.GpuDeviceId = 0;
            var mlNetModel = pipeline.Fit(dataView);

            return mlNetModel;
        }
        catch (Exception ex)
        {
            Debug.WriteLine(ex.Message);
        }
        return null;
    }

    public PredictionEngine<T, TT> GetMlNetPredictionEngine<TT>()
        where TT : class, IOnnxObjectPrediction, new()
    {
        return mlContext.Model.CreatePredictionEngine<T, TT>(mlModel);
    }

    public void SaveMLNetModel(string mlnetModelFilePath)
    {
        // Save/persist the model to a .ZIP file to be loaded by the PredictionEnginePool
        mlContext.Model.Save(mlModel, null, mlnetModelFilePath);
    }
}
