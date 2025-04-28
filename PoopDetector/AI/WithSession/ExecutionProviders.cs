namespace PoopDetector.AI.Vision;

public enum ExecutionProviders
{
    CPU,   // CPU execution provider is always available by default
    NNAPI, // NNAPI is available on Android
    CoreML, // CoreML is available on iOS/macOS
    OpenVINO, // Intel - needs verification
    CUDA
}
