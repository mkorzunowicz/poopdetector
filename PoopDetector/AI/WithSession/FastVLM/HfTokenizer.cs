using System.Text.Json;
using Microsoft.ML.Tokenizers;

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
        _tokenizer =  BpeTokenizer.Create(vocabPath, mergesPath);

        // Infer EOS/BOS from tokenizer.json special tokens; fallback to common defaults.
        //try
        //{
        //    using var doc = JsonDocument.Parse(json);
        //    var root = doc.RootElement;
        //    int eos = 2; // default guess
        //    int bos = 1; // default guess
        //    if (root.TryGetProperty("added_tokens", out var added))
        //    {
        //        foreach (var t in added.EnumerateArray())
        //        {
        //            if (t.TryGetProperty("special", out var sp) && sp.GetBoolean() &&
        //                t.TryGetProperty("content", out var content) &&
        //                t.TryGetProperty("id", out var idEl))
        //            {
        //                var c = content.GetString();
        //                if (c == "</s>") eos = idEl.GetInt32();
        //                else if (c == "<s>") bos = idEl.GetInt32();
        //            }
        //        }
        //    }
        //    _eosId = eos;
        //    _bosId = bos;
        //}
        //catch
        //{
        //    _eosId = 2;
        //    _bosId = 1;
        //}
    }

    public int[] Encode(string text)
    {
        var enc = _tokenizer.EncodeToIds(text);
        return enc.ToArray();
    }
    public string Decode(IReadOnlyList<int> tokens) => _tokenizer.Decode(tokens);
}
