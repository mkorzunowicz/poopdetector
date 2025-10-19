using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

namespace PoopDetector.AI.Vision;

internal sealed class VisionModelOptions
{
    public string? RepositoryBaseUrl { get; set; }
    public Dictionary<string, string> RemoteOverrides { get; set; } = new(StringComparer.OrdinalIgnoreCase);
    public List<string> BundledModels { get; set; } = new();

    internal string? ResolveRemoteUrl(string key, string fileName, string? fallback)
    {
        if (RemoteOverrides != null &&
            RemoteOverrides.TryGetValue(key, out var explicitUrl) &&
            !string.IsNullOrWhiteSpace(explicitUrl))
        {
            return explicitUrl;
        }

        if (!string.IsNullOrWhiteSpace(RepositoryBaseUrl))
        {
            return CombineUrl(RepositoryBaseUrl!, fileName);
        }

        return fallback;
    }

    static string CombineUrl(string baseUrl, string fileName)
    {
        if (string.IsNullOrEmpty(fileName))
        {
            return baseUrl;
        }

        if (!baseUrl.EndsWith('/'))
        {
            baseUrl += "/";
        }

        return baseUrl + fileName;
    }
}

internal static class VisionModelOptionsLoader
{
    public static VisionModelOptions Load()
    {
        try
        {
            var assembly = typeof(VisionModelManager).Assembly;
            string? resourceName = assembly
                .GetManifestResourceNames()
                .FirstOrDefault(n => n.EndsWith("appsettings.json", StringComparison.OrdinalIgnoreCase));

            if (resourceName == null)
            {
                return new VisionModelOptions();
            }

            using Stream? stream = assembly.GetManifestResourceStream(resourceName);
            if (stream == null)
            {
                return new VisionModelOptions();
            }

            var configuration = new ConfigurationBuilder()
                .AddJsonStream(stream)
                .Build();

            return configuration
                       .GetSection("VisionModels")
                       .Get<VisionModelOptions>()
                   ?? new VisionModelOptions();
        }
        catch
        {
            return new VisionModelOptions();
        }
    }
}
