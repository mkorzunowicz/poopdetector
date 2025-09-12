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
        
        // Try to find EOS and BOS tokens by encoding known patterns
        try
        {
            // Try to find special tokens by content - FastVLM uses ChatML format
            // From tokenizer_config.json:
            // "151643": "<|endoftext|>" (pad_token)
            // "151644": "<|im_start|>" 
            // "151645": "<|im_end|>" (eos_token)
            // "151646": "<image>"
            
            var eosTokens = _tokenizer.EncodeToIds("<|im_end|>");
            if (eosTokens.Count > 0) _eosId = eosTokens[0];
            
            var bosTokens = _tokenizer.EncodeToIds("<|im_start|>");
            if (bosTokens.Count > 0) _bosId = bosTokens[0];
            
            // If direct encoding didn't work, use the known token IDs from tokenizer_config.json
            if (_eosId == 0) _eosId = 151645; // <|im_end|>
            if (_bosId == 0) _bosId = 151644; // <|im_start|>
            
            Debug.WriteLine($"Using EOS ID: {_eosId} (<|im_end|>), BOS ID: {_bosId} (<|im_start|>)");
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Error finding special tokens: {ex.Message}");
            _eosId = 2;
            _bosId = 1;
        }
    }

    public int[] Encode(string text)
    {
        var enc = _tokenizer.EncodeToIds(text);
        return enc.ToArray();
    }
    public string Decode(IReadOnlyList<int> tokens) 
    {
        var decoded = _tokenizer.Decode(tokens);
        
        // According to tokenizer_config.json: "clean_up_tokenization_spaces": false
        // This means we should NOT automatically clean up Ġ and other BPE artifacts
        // However, for our UI we want readable text, so we'll clean them manually
        
        // BPE tokenizers use Ġ to represent spaces at word beginnings
        // and Ċ to represent newlines
        var cleaned = decoded.Replace("Ġ", " ").Replace("Ċ", "\n");
        
        // Remove any double spaces and trim
        while (cleaned.Contains("  ")) cleaned = cleaned.Replace("  ", " ");
        return cleaned.Trim();
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
