using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PoopDetector.AI
{
    public class AIModelManager
    {
        private static readonly Lazy<AIModelManager> _instance = new Lazy<AIModelManager>(() => new AIModelManager());
        private IYolo<IImageInputData> _model;

        private AIModelManager()
        {
        }
        public bool IsLoaded { get { return _model != null; } }
        public static AIModelManager Instance => _instance.Value;

        private IYolo<IImageInputData> LoadModel()
        {
            //return new Yolov2("TinyYolo2_model.onnx");
            //return new YoloX<TinyYoloImageInputData>(new TinyYoloXModel());
            return new YoloX<TinyYoloImageInputData>(new NanoYoloXModel());
            //return new YoloX<TinyYoloImageInputData>(new TinyYoloXPoopModel());
            //return new YoloX<SmallYoloImageInputData>(new SmallYoloXModel());
        }
        public enum ModelTypes
        {
            //TinyYoloX,
            //TinyYoloXPoop,
            //NanoYoloXPoopCroppedLast,
            //NanoYoloXPoopCroppedBest,
            //TinyYoloXPoopCroppedLast,
            TinyYoloXPoopCroppedBest,
            NanoYoloX,
            //NanoYoloXPoop81,
            SmallYoloX,
            //TinyYoloV2,
            //NanoYoloXPoopCroppedOnlyLast,
            NanoYoloXPoopCroppedOnlyBest,
            //NanoYoloXPoopCroppedOnly700Last,
            NanoYoloXPoopCroppedOnly80Best
        }
        public async Task ChangeModelAsync(ModelTypes type)
        {
            _model = await Task.Run<IYolo<IImageInputData>>(() =>
            {
                return type switch
                {
                    ModelTypes.NanoYoloX => new YoloX<TinyYoloImageInputData>(new NanoYoloXModel()),
                    //ModelTypes.SmallYoloX => new YoloX<SmallYoloImageInputData>(new SmallYoloXModel()),
                    //ModelTypes.TinyYoloX => new YoloX<TinyYoloImageInputData>(new TinyYoloXModel()),
                    //ModelTypes.NanoYoloXPoop81 => new YoloX<TinyYoloImageInputData>(new NanoYoloXAndPoopModel()),
                    //ModelTypes.TinyYoloXPoop => new YoloX<TinyYoloImageInputData>(new TinyYoloXPoopModel()),
                    //ModelTypes.NanoYoloXPoopCroppedLast => new YoloX<TinyYoloImageInputData>(new TinyYoloXPoopModel("yolox_nano_poop_cropped_last.onnx")),
                    //ModelTypes.NanoYoloXPoopCroppedBest => new YoloX<TinyYoloImageInputData>(new TinyYoloXPoopModel("yolox_nano_poop_cropped_best.onnx")),
                    //ModelTypes.TinyYoloXPoopCroppedLast => new YoloX<TinyYoloImageInputData>(new TinyYoloXPoopModel("yolox_tiny_poop_cropped_last.onnx")),
                    ModelTypes.TinyYoloXPoopCroppedBest => new YoloX<TinyYoloImageInputData>(new TinyYoloXPoopModel("yolox_tiny_poop_cropped_best.onnx")),
                    //ModelTypes.NanoYoloXPoopCroppedOnlyLast => new YoloX<TinyYoloImageInputData>(new TinyYoloXPoopModel("yolox_nano_poop_cropped_only_last.onnx")),
                    ModelTypes.NanoYoloXPoopCroppedOnlyBest => new YoloX<TinyYoloImageInputData>(new TinyYoloXPoopModel("yolox_nano_poop_cropped_only_best.onnx")),
                    //ModelTypes.NanoYoloXPoopCroppedOnly700Last => new YoloX<SmallYoloImageInputData>(new TinyYoloXPoopModel("yolox_nano_poop_cropped_only_700_last.onnx", 640, 640)),
                    ModelTypes.NanoYoloXPoopCroppedOnly80Best => new YoloX<SmallYoloImageInputData>(new TinyYoloXPoopModel("yolox_nano_poop_cropped_only_700_best.onnx", 640, 640)),
                    //ModelTypes.TinyYoloV2 => new Yolov2<TinyYoloImageInputData>("TinyYolo2_model.onnx"),

                    //_ => new YoloX<TinyYoloImageInputData>(new TinyYoloXModel()),
                    _ => new YoloX<TinyYoloImageInputData>(new NanoYoloXModel()),
                };
            });
        }
        public async Task LoadModelAsync()
        {

            _model = await Task.Run(LoadModel);
        }

        public IYolo<IImageInputData> GetModel()
        {
            return _model;
        }
    }
}
