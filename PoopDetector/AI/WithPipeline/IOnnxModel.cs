using Microsoft.ML.Data;
using PoopDetector.Views;
using System.Diagnostics;

namespace PoopDetector.AI;

public interface IOnnxObjectPrediction
{
    float[] PredictedLabels { get; set; }
}
public interface IOnnxModel
{
    public int InputHeight { get; }
    public int InputWidth { get; }
    string ModelPath { get; }

    // To check Model input and output parameter names, you can
    // use tools like Netron: https://github.com/lutzroeder/netron
    // Or read the Exception message which states what is expected
    string ModelInput { get; }
    string ModelOutput { get; }

    //string[] Labels { get; }
    List<(string, System.Drawing.Color)> ColormapList { get; }
    (float, float)[] Anchors { get; }


    public abstract class AbstractOnnxModel : IOnnxModel
    {
        public abstract IImageInputData GetInputData(MLImage image);
        public int InputHeight { get; internal set; }
        public int InputWidth { get; internal set; }
        public string ModelPath { get; internal set; }
        public string ModelInput { get; internal set; }
        public string ModelOutput { get; internal set; }
        public List<(string, System.Drawing.Color)> ColormapList { get; internal set; }
        public (float, float)[] Anchors { get; internal set; }

        public void Load(string modelName)
        {
            var resourcePrefix = "PoopDetector.Resources.ai.";
            var assembly = System.Reflection.IntrospectionExtensions.GetTypeInfo(typeof(CameraPage)).Assembly;

            ModelPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), modelName);
            if (File.Exists(ModelPath))
            {
                // File.Delete(path);
            }
            else
            {
                try
                {
                    using var fs = File.Create(ModelPath);
                    using var stream = assembly.GetManifestResourceStream(resourcePrefix + modelName);
                    stream.Seek(0, SeekOrigin.Begin);
                    stream.CopyTo(fs);
                    stream.Close();
                }
                catch (Exception ex) {
                    Debug.WriteLine($"Problems loading model: {modelName}");
                Debug.WriteLine(ex.ToString());
                }
            }
        }
    }
}
