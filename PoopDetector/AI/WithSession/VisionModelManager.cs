using PoopDetector.AI.Vision.YoloX;

namespace PoopDetector.AI.Vision
{
    public class VisionModelManager
    {
        private static readonly Lazy<VisionModelManager> _instance = new Lazy<VisionModelManager>(() => new VisionModelManager());
        private IVision _poopModel;
        private IVision _yoloModel;
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
                return type switch
                {
                    ModelTypes.YoloxNanoPoopCroppedOnlyBest => new YoloXNanoPoop(),
                    ModelTypes.YoloxNanoPoopMixed => new YoloXNanoPoop("yolox_nano_poop_mixed.onnx"),
                    ModelTypes.YoloxTinyPoop => new YoloXNano("yolox_tiny_poop.onnx"),
                    ModelTypes.YoloxNano => new YoloXNano(),
                    ModelTypes.YoloxSmall => new YoloX.YoloX(),
                    _ => new YoloXNanoPoop(),
                };
            });
        }
        private VisionModelManager()
        {
        }
        public bool IsLoaded { get { return CurrentModel != null; } }
        // public bool IsLoaded { get { return _poopModel != null && _yoloModel != null; } }
        public static VisionModelManager Instance => _instance.Value;

        public IVision PoopModel => _poopModel;
        public IVision YoloModel => _yoloModel;
        public MobileSam.MobileSam MobileSam => _samModel;
        private void LoadModel()
        {
            _model = new YoloXNanoPoop();

            //_poopModel = new YoloXDoubleNanoPoop();
            // _poopModel = new YoloXNanoPoop();
            // _yoloModel = new YoloXNano();
            _samModel = new MobileSam.MobileSam();
        }
        public async Task LoadModelAsync()
        {
            await Task.Run(LoadModel);
        }
    }
}
