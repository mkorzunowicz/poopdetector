using System.Text.Json;
using Microsoft.ML.Tokenizers;
using System.Diagnostics;

namespace PoopDetector.AI.Vision.FastVLM;

/// Thin wrapper around Microsoft.ML.Tokenizers loading from tokenizer.json using Tokenizer.FromFile.
internal sealed class HfTokenizer : ITokenizer
{
    readonly Tokenizer _tokenizer;
    readonly int _eosId;
    readonly int _bosId;

    public int EosId => _eosId;
    public int BosId => _bosId;

    public HfTokenizer(string vocabPath, string mergesPath)
    {
        _tokenizer = BpeTokenizer.Create(vocabPath, mergesPath);
        
        // Use the known token IDs from FastVLM tokenizer_config.json
        // Don't try to encode these as the BPE tokenizer won't recognize them as single tokens
        _eosId = 151645; // <|im_end|>
        _bosId = 151644; // <|im_start|>
        
        Debug.WriteLine($"Using EOS ID: {_eosId} (<|im_end|>), BOS ID: {_bosId} (<|im_start|>)");
    }

    public int[] Encode(string text)
    {
        // Handle special tokens manually since they're not in the BPE vocab
        var processedText = text;
        var specialTokens = new Dictionary<string, int>
        {
            {"<|im_start|>", 151644},
            {"<|im_end|>", 151645},
            {"<image>", 151646},
            {"<|endoftext|>", 151643}
        };
        
        var result = new List<int>();
        var currentPos = 0;
        
        while (currentPos < processedText.Length)
        {
            bool foundSpecialToken = false;
            
            // Check for special tokens at current position
            foreach (var (token, id) in specialTokens)
            {
                if (processedText.Substring(currentPos).StartsWith(token))
                {
                    result.Add(id);
                    currentPos += token.Length;
                    foundSpecialToken = true;
                    break;
                }
            }
            
            if (!foundSpecialToken)
            {
                // Find the end of regular text (until next special token or end)
                int nextSpecialPos = processedText.Length;
                foreach (var (token, _) in specialTokens)
                {
                    int pos = processedText.IndexOf(token, currentPos);
                    if (pos >= 0 && pos < nextSpecialPos)
                    {
                        nextSpecialPos = pos;
                    }
                }
                
                // Encode the regular text portion
                if (nextSpecialPos > currentPos)
                {
                    var regularText = processedText.Substring(currentPos, nextSpecialPos - currentPos);
                    var regularTokens = _tokenizer.EncodeToIds(regularText);
                    result.AddRange(regularTokens);
                    currentPos = nextSpecialPos;
                }
                else
                {
                    // Single character that's not part of a special token
                    var singleChar = processedText.Substring(currentPos, 1);
                    var singleTokens = _tokenizer.EncodeToIds(singleChar);
                    result.AddRange(singleTokens);
                    currentPos++;
                }
            }
        }
        
        return result.ToArray();
    }
    public string Decode(IReadOnlyList<int> tokens) 
    {
        // Handle special tokens manually
        var specialTokens = new Dictionary<int, string>
        {
            {151644, "<|im_start|>"},
            {151645, "<|im_end|>"},
            {151646, "<image>"},
            {151643, "<|endoftext|>"}
        };
        
        var result = new List<string>();
        var regularTokens = new List<int>();
        
        foreach (var token in tokens)
        {
            if (specialTokens.ContainsKey(token))
            {
                // First decode any accumulated regular tokens
                if (regularTokens.Count > 0)
                {
                    var decoded = _tokenizer.Decode(regularTokens);
                    var cleaned = decoded.Replace("Ġ", " ").Replace("Ċ", "\n");
                    result.Add(cleaned);
                    regularTokens.Clear();
                }
                
                // Add the special token
                result.Add(specialTokens[token]);
            }
            else
            {
                // Accumulate regular tokens
                regularTokens.Add(token);
            }
        }
        
        // Decode any remaining regular tokens
        if (regularTokens.Count > 0)
        {
            var decoded = _tokenizer.Decode(regularTokens);
            var cleaned = decoded.Replace("Ġ", " ").Replace("Ċ", "\n");
            result.Add(cleaned);
        }
        
        var finalResult = string.Join("", result);
        
        // Remove any double spaces and trim
        while (finalResult.Contains("  ")) finalResult = finalResult.Replace("  ", " ");
        return finalResult.Trim();
    }
    
    public bool ContainsImageToken(int[] tokens)
    {
        // Check if the token sequence contains the image token (151646)
        return tokens.Contains(151646);
    }
    
    public int[] EncodeWithImageToken(string text)
    {
        // Try to encode the text and manually replace <image> with the correct token
        var tokens = Encode(text);
        var result = new List<int>();
        
        // Convert token sequence back to check for <image> text
        for (int i = 0; i < tokens.Length; i++)
        {
            var tokenText = _tokenizer.Decode(new[] { tokens[i] });
            if (tokenText.Contains("<image>"))
            {
                // Replace with the actual image token
                result.Add(151646);
            }
            else
            {
                result.Add(tokens[i]);
            }
        }
        
        return result.ToArray();
    }
}
