using Microsoft.ML;
using Microsoft.ML.Data;

namespace PoopDetector.AI
{
    public class YoloXPrediction : IOnnxObjectPrediction
    {
        [ColumnName("output")]
        public float[] PredictedLabels { get; set; }
    }
    public interface IYolo<T>
    {
        IOnnxModel Model { get; }
        Task<List<BoundingBox>> DetectObjects(T imageInputData);
        IImageInputData GetInputData(MLImage image);
    }
    public class YoloX<T> : IYolo<IImageInputData> where T : class, IImageInputData, new()
    {
        private YoloXOutputParser outputParser;
        private PredictionEngine<T, YoloXPrediction> predictionEngine;
        public YoloX(IOnnxModel model)
        {
            Model = model;
            var modelConfigurator = new OnnxModelConfigurator<T>(model);

            outputParser = new YoloXOutputParser(model);
            predictionEngine = modelConfigurator.GetMlNetPredictionEngine<YoloXPrediction>();
        }

        public IOnnxModel Model { get; }

        public async Task<List<BoundingBox>> DetectObjects(IImageInputData imageInputData)
        {
            var labels = await Task.Run(() => predictionEngine?.Predict((T)imageInputData).PredictedLabels);
            var boundingBoxes = outputParser.ParseOutputs(labels);
            return boundingBoxes;
        }

        public IImageInputData GetInputData(MLImage image)
        {
            return new T { Image = image };
        }
    }
}
