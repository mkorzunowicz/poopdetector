namespace PoopDetector.AI.Vision.FastVLM;

/// Abstraction over tokenizer so we can plug in Microsoft.ML.Tokenizers
/// while still compiling if the package is not present at design time.
internal interface ITokenizer
{
    int[] Encode(string text);
    string Decode(IReadOnlyList<int> tokens);
    int EosId { get; }
    int BosId { get; }
}

