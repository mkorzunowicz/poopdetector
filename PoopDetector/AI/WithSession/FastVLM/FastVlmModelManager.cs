using PoopDetector.Services;

namespace PoopDetector.AI.Vision.FastVLM;

/// <summary>
/// Handles FastVLM model downloading, variant management, and file path resolution.
/// Extracted from VisionModelManager to keep FastVLM-specific logic together.
/// </summary>
public static class FastVlmModelManager
{
    /// <summary>
    /// Downloads all required FastVLM model files for the specified variant
    /// </summary>
    public static async Task<string> DownloadVariantAsync(string variant, IProgress<double> progress, CancellationToken ct)
    {
        string root = "https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX/resolve/main/";
        
        var (visionSuffix, embedSuffix, decoderSuffix) = GetVariantSuffixes(variant);

        // Build file names - keep variant-specific names for all files
        string visionFile = $"vision_encoder{visionSuffix}.onnx";
        string embedFile = $"embed_tokens{embedSuffix}.onnx";
        string decoderFile = $"decoder_model_merged{decoderSuffix}.onnx";
        
        // Download files with variant-specific names to avoid overwriting
        const int totalFiles = 6;
        double fileProgress = 1.0 / totalFiles;
        
        // Download vision encoder with variant-specific name
        _ = await ModelCache.GetAsync(root + "onnx/" + visionFile, visionFile, 
            new Progress<double>(p => progress.Report(0 * fileProgress + p * fileProgress)), ct);
            
        _ = await ModelCache.GetAsync(root + "onnx/" + embedFile, embedFile, 
            new Progress<double>(p => progress.Report(1 * fileProgress + p * fileProgress)), ct);
            
        string dec = await ModelCache.GetAsync(root + "onnx/" + decoderFile, decoderFile, 
            new Progress<double>(p => progress.Report(2 * fileProgress + p * fileProgress)), ct);
            
        _ = await ModelCache.GetAsync(root + "merges.txt", "merges.txt", 
            new Progress<double>(p => progress.Report(3 * fileProgress + p * fileProgress)), ct);
            
        _ = await ModelCache.GetAsync(root + "vocab.json", "vocab.json", 
            new Progress<double>(p => progress.Report(4 * fileProgress + p * fileProgress)), ct);
            
        _ = await ModelCache.GetAsync(root + "tokenizer.json", "tokenizer.json", 
            new Progress<double>(p => progress.Report(5 * fileProgress + p * fileProgress)), ct);
            
        progress.Report(1.0);
        return dec; // return any; wrapper will resolve others from AppDataDirectory
    }

    /// <summary>
    /// Gets the file paths for all FastVLM model components based on variant
    /// </summary>
    public static (string visionEncoderPath, string embedPath, string decoderPath) GetFilePaths(string variant)
    {
        var (visionSuffix, embedSuffix, decoderSuffix) = GetVariantSuffixes(variant);

        string visionFile = $"vision_encoder{visionSuffix}.onnx";
        string embedFile = $"embed_tokens{embedSuffix}.onnx";
        string decoderFile = $"decoder_model_merged{decoderSuffix}.onnx";
        
        return (Path.Combine(FileSystem.Current.AppDataDirectory, visionFile),
                Path.Combine(FileSystem.Current.AppDataDirectory, embedFile),
                Path.Combine(FileSystem.Current.AppDataDirectory, decoderFile));
    }

    /// <summary>
    /// Determines the file suffixes for model files based on variant type
    /// </summary>
    private static (string visionSuffix, string embedSuffix, string decoderSuffix) GetVariantSuffixes(string variant)
    {
        // Supports: "", q4f16, fp16, int8, q4, bnb4, quantized
        return variant switch
        {
            "" => ("", "", ""), // Full precision (original)
            
            "mixed" or "q4f16" => ("_q4f16", "_q4f16", "_q4f16"), // Mixed precision (default)
            
            "fp16" or "f16" => ("_fp16", "_fp16", "_fp16"), // All FP16
            
            "int8" => ("_fp16", "_int8", "_int8"), // All INT8 (vision encoder stays fp16)
            
            "q4" => ("_q4", "_q4", "_q4"), // 4-bit quantized
            
            "bnb4" => ("_bnb4", "_int8", "_int8"), // BitsAndBytes 4-bit (smallest working combo)
            
            "quantized" => ("_quantized", "_quantized", "_quantized"), // Generic quantized (fastest)
            
            _ => throw new ArgumentException(
                $"Unknown FastVLM variant: {variant}. " +
                $"Supported variants: '', 'mixed', 'q4f16', 'fp16', 'int8', 'q4', 'bnb4', 'quantized'")
        };
    }

    /// <summary>
    /// Get all available FastVLM model variants
    /// </summary>
    public static string[] GetAvailableVariants()
    {
        return new string[] 
        { 
            "",        // Full precision (original)
            "q4f16",   // Mixed precision (default)
            "fp16",    // All FP16
            "int8",    // All INT8
            "q4",      // 4-bit quantized
            "bnb4",    // BitsAndBytes 4-bit
            "quantized" // Generic quantized
        };
    }

    /// <summary>
    /// Get human-readable descriptions for FastVLM variants
    /// </summary>
    public static Dictionary<string, string> GetVariantDescriptions()
    {
        return new Dictionary<string, string>
        {
            [""] = "Full Precision (Original) - Highest quality, largest size",
            ["q4f16"] = "Q4F16 Mixed Precision - Good balance of speed/quality (Default)",
            ["fp16"] = "Half Precision (FP16) - Faster inference, moderate size",
            ["int8"] = "8-bit Integer - Fast inference, smaller size",
            ["q4"] = "4-bit Quantized - Very fast, compact size",
            ["bnb4"] = "BitsAndBytes 4-bit - Optimized 4-bit quantization",
            ["quantized"] = "Generic Quantized - General purpose quantization"
        };
    }
}
