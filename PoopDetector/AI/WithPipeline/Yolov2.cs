using Microsoft.ML;
using Microsoft.ML.Data;
using PoopDetector.Views;
using System.Diagnostics;

namespace PoopDetector.AI
{
    public class TinyYoloPrediction : IOnnxObjectPrediction
    {
        [ColumnName("grid")]
        public float[] PredictedLabels { get; set; }
    }

    public class Yolov2<T> : IYolo<IImageInputData> where T : class, IImageInputData, new()
    {
        private OnnxOutputParser outputParser;
        private PredictionEngine<T, TinyYoloPrediction> tinyYoloPredictionEngine;
        public IOnnxModel Model { get; }

        public Yolov2(string modelFileName)
        {
            var resourcePrefix = "PoopDetector.Resources.ai.";
            var assembly = System.Reflection.IntrospectionExtensions.GetTypeInfo(typeof(CameraPage)).Assembly;

            var path = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), modelFileName);
            if (File.Exists(path))
                Debug.WriteLine("Model already copied");
            else
            {
                using var fs = File.Create(path);
                using var stream = assembly.GetManifestResourceStream(resourcePrefix + modelFileName);
                stream.Seek(0, SeekOrigin.Begin);
                stream.CopyTo(fs);
            }

            var modelsPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), modelFileName);
            Model = new TinyYoloModel(modelsPath);


            var modelConfigurator = new OnnxModelConfigurator<T>(Model);

            outputParser = new OnnxOutputParser(Model);
            tinyYoloPredictionEngine = modelConfigurator.GetMlNetPredictionEngine<TinyYoloPrediction>();
        }
        public async Task<List<BoundingBox>> DetectObjects(IImageInputData imageInputData)
        {
            var labels = await Task.Run(()=>tinyYoloPredictionEngine?.Predict((T)imageInputData).PredictedLabels);
            var boundingBoxes = outputParser.ParseOutputs(labels);
            var filteredBoxes = outputParser.FilterBoundingBoxes(boundingBoxes, 5, 0.5f);
            return filteredBoxes;
        }
        public IImageInputData GetInputData(MLImage image)
        {
            return new T { Image = image };
        }
    }
}
