// File: Converters/FileImageConverter.cs
using System;
using System.Globalization;
using Microsoft.Maui.Controls;

namespace PoopDetector.Converters;

/// <summary>
/// Turns an absolute file path into an <see cref="ImageSource"/>.
/// Works for both Android and iOS local-storage files.
/// </summary>
public sealed class FileImageConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture) =>
        value is string path ? ImageSource.FromFile(path) : null;

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}
