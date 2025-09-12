using Microsoft.ML.Tokenizers;
using System.Text;

namespace PoopDetector.AI.Vision.FastVLM;

/// GPT-2 BPE with manual ChatML specials and a "byte-clean" decoder to remove Ċ/Ġ/byte-glyphs.
internal sealed class HfTokenizer 
{
    // ChatML / Qwen2 specials
    public const int EndOfTextId = 151643;   // <|endoftext|>
    public const int ImStartId = 151644;   // <|im_start|>
    public const int ImEndId = 151645;   // <|im_end|>
    public const int ImageId = 151646;   // <image>

    private readonly Tokenizer _bpe;

    public int EosId => ImEndId;
    public int BosId => ImStartId;

    public HfTokenizer(string vocabPath, string mergesPath)
    {
        // Keep the (vocab, merges) order you verified.
        _bpe = BpeTokenizer.Create(vocabPath, mergesPath);
    }

    // Plain encode (no specials processed)
    public int[] Encode(string text) => _bpe.EncodeToIds(text).ToArray();

    // ChatML prompt with a SINGLE <image> (to be expanded later)
    public List<int> TokenizeChatWithSingleImage(string system, string user, bool addGenerationPrompt)
    {
        var ids = new List<int>();
        // <|im_start|>system\n{system}<|im_end|>\n
        ids.Add(ImStartId);
        ids.AddRange(EncodePlain("system\n"));
        ids.AddRange(EncodePlain(system));
        ids.Add(ImEndId);
        ids.AddRange(EncodePlain("\n"));
        // <|im_start|>user\n<image>{user}<|im_end|>\n
        ids.Add(ImStartId);
        ids.AddRange(EncodePlain("user\n"));
        ids.Add(ImageId);
        ids.AddRange(EncodePlain(user.Replace("<image>", "")));
        ids.Add(ImEndId);
        ids.AddRange(EncodePlain("\n"));
        if (addGenerationPrompt)
        {
            // <|im_start|>assistant\n
            ids.Add(ImStartId);
            ids.AddRange(EncodePlain("assistant\n"));
        }
        return ids;
    }

    // Expand the single <image> marker to N contiguous image tokens
    public List<int> ExpandSingleImageToken(List<int> input, int nImageTokens)
    {
        int idx = input.FindIndex(t => t == ImageId);
        if (idx < 0) throw new InvalidOperationException("No <image> token found in the sequence.");
        var outIds = new List<int>(input.Count - 1 + nImageTokens);
        outIds.AddRange(input.Take(idx));
        for (int i = 0; i < nImageTokens; i++) outIds.Add(ImageId);
        outIds.AddRange(input.Skip(idx + 1));
        return outIds;
    }

    // Locate the contiguous image block
    public (int start, int length) FindImageTokenBlock(List<int> tokens)
    {
        int start = tokens.FindIndex(t => t == ImageId);
        if (start < 0) return (-1, 0);
        int i = start;
        while (i < tokens.Count && tokens[i] == ImageId) i++;
        return (start, i - start);
    }

    // Default decode (kept for compatibility)
    public string Decode(IReadOnlyList<int> tokens) => _bpe.Decode(tokens);

    // Clean decode: skip specials and byte-clean GPT-2 glyphs (Ġ → space, Ċ → newline, etc.)
    public string DecodeClean(IReadOnlyList<int> tokens)
    {
        var filtered = tokens.Where(t => t != EndOfTextId && t != ImStartId && t != ImEndId && t != ImageId).ToArray();
        if (filtered.Length == 0) return string.Empty;

        // Start with ML.Tokenizers decode (fast), then byte-clean common GPT-2 artifacts.
        var s = _bpe.Decode(filtered);

        // Common whitespace artifacts
        s = s.Replace("Ġ", " ").Replace("Ċ", "\n");

        // Byte-glyph cleanup: map Latin-1 lookalikes back down to bytes when they appear in pairs like "Ã©" → "é"
        // This simple pass fixes most "Ã", "Â", etc. cases without a full GPT-2 byte map.
        s = FixUtf8Artifacts(s);

        // Collapse multiple spaces
        while (s.Contains("  ")) s = s.Replace("  ", " ");

        return s.Trim();
    }

    // ---- internals ----

    private int[] EncodePlain(string text) => _bpe.EncodeToIds(text).ToArray();

    /// <summary>
    /// Heuristic UTF-8 artifact fixer (handles common sequences like "Ã©" → "é", "Â·" → "·", etc.).
    /// This avoids shipping a 256-entry GPT-2 byte table and is sufficient for normal English outputs.
    /// </summary>
    private static string FixUtf8Artifacts(string input)
    {
        // Quick bail-out if nothing suspicious
        if (input.IndexOf('Ã') < 0 && input.IndexOf('Â') < 0) return input;

        // Interpret the string as if it contained mis-decoded UTF-8 sequences.
        // Convert to bytes with Latin-1, then back to UTF-8.
        // This trick turns "Ã©" into the single Unicode "é", etc.
        var latin1 = Encoding.GetEncoding("ISO-8859-1");
        var bytes = latin1.GetBytes(input);
        return Encoding.UTF8.GetString(bytes);
    }
}
