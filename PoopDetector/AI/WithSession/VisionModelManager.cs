using PoopDetector.AI.Vision.YoloX;

namespace PoopDetector.AI.Vision
{
    public class VisionModelManager
    {
        private static readonly Lazy<VisionModelManager> _instance = new Lazy<VisionModelManager>(() => new VisionModelManager());
        private IVision _poopModel;
        private IVision _yoloModel;
        private MobileSam.MobileSam _samModel;

        private VisionModelManager()
        {
        }
        //public bool IsLoaded { get { return _poopModel != null; } }
        public bool IsLoaded { get { return _poopModel != null && _yoloModel != null; } }
        public static VisionModelManager Instance => _instance.Value;

        public IVision PoopModel => _poopModel;
        public IVision YoloModel => _yoloModel;
        public MobileSam.MobileSam MobileSam => _samModel;
        private void LoadModel()
        {
            //_poopModel = new YoloXDoubleNanoPoop();
            _poopModel = new YoloXNanoPoop();
            _yoloModel = new YoloXNano();
            _samModel = new MobileSam.MobileSam();
        }
        public async Task LoadModelAsync()
        {
            await Task.Run(LoadModel);
        }
    }
}
