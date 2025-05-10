using PoopDetector.AI.Vision.YoloX;

namespace PoopDetector.AI.Vision
{
    public class VisionModelManager
    {
        private static readonly Lazy<VisionModelManager> _instance = new Lazy<VisionModelManager>(() => new VisionModelManager());
        private MobileSam.MobileSam _samModel;

        public IVision CurrentModel => _model;
        private IVision _model;
        public enum ModelTypes
        {
            YoloxNanoPoopMixed,
            YoloxTinyPoop,
            YoloxNanoPoopCroppedOnlyBest,
            YoloxNano,
            YoloxSmall,
        }
        public async Task ChangeModelAsync(ModelTypes type)
        {
            _model = await Task.Run<IVision>(() =>
            {
                var poopAndYolo = YoloXColormap.ColormapList.ToList();
                poopAndYolo.AddRange(YoloXColormap.PoopList);
                return type switch
                {
                    ModelTypes.YoloxNanoPoopCroppedOnlyBest => new YoloX.YoloX("yolox_nano_poop_cropped_only_best.onnx", 416,416,YoloXColormap.PoopList),
                    ModelTypes.YoloxNanoPoopMixed => new YoloX.YoloX("yolox_nano_poop_mixed.onnx",416,416,poopAndYolo),
                    ModelTypes.YoloxTinyPoop => new YoloX.YoloX("yolox_tiny_poop.onnx",416,416,YoloXColormap.PoopList),
                    ModelTypes.YoloxNano => new YoloX.YoloX("yolox_nano.onnx",416,416,YoloXColormap.ColormapList),
                    // ModelTypes.YoloxSmal => new YoloX.YoloX("yolox_s_export_op11.onnx",416,416,YoloXColormap.PoopList),
                    ModelTypes.YoloxSmall => new YoloX.YoloX("yolox_s_export_op11.onnx",640,640, YoloXColormap.ColormapList),
                    _ => new YoloX.YoloX("yolox_s_export_op11.onnx",640,640, YoloXColormap.ColormapList),
                };
            });
        }
        private VisionModelManager()
        {
        }
        public bool IsLoaded { get { return CurrentModel != null; } }
        // public bool IsLoaded { get { return _poopModel != null && _yoloModel != null; } }
        public static VisionModelManager Instance => _instance.Value;

        public MobileSam.MobileSam MobileSam => _samModel;
        private void LoadModel()
        {
            _model = new YoloX.YoloX("yolox_nano_poop_cropped_only_best.onnx", 416,416,YoloXColormap.PoopList);

            _samModel = new MobileSam.MobileSam();
        }
        public async Task LoadModelAsync()
        {
            await Task.Run(LoadModel);
        }
    }
}
