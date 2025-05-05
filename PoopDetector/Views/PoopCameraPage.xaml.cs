using Camera.MAUI;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Storage;
using Microsoft.ML.Data;
using PoopDetector.AI;
using PoopDetector.AI.Vision;
using PoopDetector.AI.Vision.Processing;
using PoopDetector.ViewModel;
using SkiaSharp;
using SkiaSharp.Views.Maui;
using System.Diagnostics;
using System.Drawing;
using System.IO;

namespace PoopDetector.Views
{
    public partial class PoopCameraPage : ContentPage
    {
        private PoopCameraViewModel _viewModel;

        bool playing = false;
        bool debug = true;

        public PoopCameraPage()
        {
            InitializeComponent();

            _viewModel = new PoopCameraViewModel(cameraView);
            BindingContext = _viewModel;
            cameraView.CamerasLoaded += CameraView_CamerasLoaded;
            _viewModel.SelectedCameraChanged += async (camera) => await ChangeCameraAsync(camera);
            _viewModel.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(PoopCameraViewModel.SamResultReady))
                    if (_viewModel.SamResultReady)
                    Dispatcher.DispatchAsync(async () =>
                        {
                            maskCanvasView.InvalidateSurface();
                        });                        
            };
        }

        // handle taps on the frozen picture
        private async void OnFrozenTapped(object sender, TappedEventArgs e)
        {
            if (_viewModel.CurrentPrediction == null) return;

            // position inside the image control
            var p = e.GetPosition(frozenImage);
            if (p == null) return;
            if (p.Value.X < 0 || p.Value.Y < 0) return;

            var enc = VisionModelManager.Instance.MobileSam.ImageProcessor
                          .GetEncoderSize(new Microsoft.Maui.Graphics.Size(_viewModel.CurrentPrediction.OriginalWidth, _viewModel.CurrentPrediction.OriginalHeight));
            // map from control space -> original pixel space
            var sx = enc.Width / frozenImage.Width;
            var sy = enc.Height / frozenImage.Height;

            double scale = Math.Max(sx, sy);
            // TODO: fix the outside image frame click
            var x_enc = p.Value.X * scale;
            var y_enc = p.Value.Y * scale;

            //Debug.WriteLine($"scale: ({sx}, {sy} ) cam: ({p.Value.X},{p.Value.Y}) mask:({frozenImage.Width},{frozenImage.Height})");

            await _viewModel.CurrentPrediction.RunSamDecode(
                    new Microsoft.Maui.Graphics.PointF((float)x_enc, (float)y_enc));

            _viewModel.SamResultReady = true;          // trigger repaint
        }
        private void OnMaskPaintSurface(object sender, SKPaintSurfaceEventArgs e)
        {
            var canvas = e.Surface.Canvas;
            var pr = _viewModel.CurrentPrediction;
            canvas.Clear();
            if (pr?.MaskBitmaps?.Count == 0)
            {
                _viewModel.SamResultReady = false;
                return;
            }

            var mask = pr.MaskBitmaps[0];
            //Debug.WriteLine("masks: " + pr.MaskBitmaps.Count);
            if (mask.IsEmpty) return;
            if (mask.IsNull) return;
            if (!mask.ReadyToDraw) return;

            float scaleImg = Math.Min(
                e.Info.Width / (float)pr.OriginalWidth,
                e.Info.Height / (float)pr.OriginalHeight);
            float offX = (e.Info.Width - pr.OriginalWidth * scaleImg) / 2f;
            float offY = (e.Info.Height - pr.OriginalHeight * scaleImg) / 2f;

            canvas.Translate(offX, offY);

            float sx = (float)e.Info.Width / mask.Width;
            float sy = (float)e.Info.Height / mask.Height;
            canvas.Scale(Math.Min(sx, sy));

            //Debug.WriteLine($"scale: ({sx}, {sy} ) cam: ({e.Info.Width},{e.Info.Height}) mask:({maskBmp.Width},{maskBmp.Height})");
            SKPaint fill = new SKPaint { IsStroke = false, Color = SKColors.Blue.WithAlpha(0x80) };
            try
            {
                // Still throws sometimes on Windows screen resize
                canvas.DrawBitmap(mask, 0, 0, fill);
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
            }

            canvas.Restore();

            if (pr?.Polygons?.Count > 0)
            {
                // We need to scale the polygons back to the mask size, as we alerady scaled it to the image
                float sx2 = (float)mask.Width / pr.OriginalWidth;
                float sy2 = (float)mask.Height / pr.OriginalHeight;

                using var polyStroke = new SKPaint
                {
                    Color = SKColors.Red,
                    StrokeWidth = 2,
                    Style = SKPaintStyle.Stroke,
                    IsAntialias = true
                };

                using var polyFill = new SKPaint
                {
                    Color = SKColors.Blue.WithAlpha(80),   // translucent fill
                    Style = SKPaintStyle.Fill,
                    IsAntialias = true
                };
                //canvas.Scale(Math.Min(sx, sy));
                foreach (var flat in pr?.Polygons)
                {
                    //var pth = BuildPath(flat, 0, 0, 1);
                    var pth = BuildPath(flat, 0, 0, Math.Min(sx2, sy2));
                    if (pth == null) continue;

                    canvas.DrawPath(pth, polyFill);   // interior
                    canvas.DrawPath(pth, polyStroke); // outline
                }
            }

            _viewModel.SamResultReady = false;
        }
        private async Task ChangeCameraAsync(CameraInfo newCamera)
        {
            width = 0; height = 0;
            if (playing)
                await cameraView.StopCameraAsync();

            cameraView.Camera = newCamera;
            playing = (await cameraView.StartCameraAsync()) == CameraResult.Success;

        }

        protected override async void OnAppearing()
        {
            base.OnAppearing();

            if (!playing)
            {
                playing = (await cameraView.StartCameraAsync()) == CameraResult.Success;
            }

            //if (playing)
            //{
            StartPredictionLoop();
            //}
        }

        protected override void OnDisappearing()
        {
            base.OnDisappearing();
            playing = false;
        }

        private async void CameraView_CamerasLoaded(object sender, EventArgs e)
        {
            // Clear and re-populate the ViewModel's Cameras
            _viewModel.Cameras.Clear();
            if (cameraView.Cameras.Count == 0)
            {
                // no cameras => the �No cameras available� label will show from the VM
                return;
            }

            foreach (var cam in cameraView.Cameras)
            {
                _viewModel.Cameras.Add(cam);
            }

            // If there's a "back" camera, pick that; otherwise pick the first
            var backCamera = cameraView.Cameras.FirstOrDefault(c => c.Position == CameraPosition.Back);
            if (backCamera != null)
                _viewModel.SelectedCamera = backCamera;
            else
                _viewModel.SelectedCamera = _viewModel.Cameras.First();

        }

        private async void StartPredictionLoop()
        {
            Debug.WriteLine("STart prediction loop.");
            await Task.Factory.StartNew(async () =>
            {
                // Wait until the model is loaded and playing is true
                while (!playing || !VisionModelManager.Instance.IsLoaded)
                {
                    await Task.Delay(10);
                }
                Stopwatch loopStopwatch = Stopwatch.StartNew();
                int loopCount = 0;

                while (playing)
                {
                    try
                    {
                        if (_viewModel.PausePredictions)
                        {
                            await Task.Delay(100);
                            continue;
                        }

                        var stream = cameraView.GetSnapShotStream(Camera.MAUI.ImageFormat.JPEG);

                        // var stream = await cameraView.TakePhotoAsync(Camera.MAUI.ImageFormat.JPEG);
                        if (stream == null || !stream.CanRead)
                        {
                            await Task.Delay(100);
                            continue;
                        }

                        ////testing fixed image:
                        //// add the file to Resources\Raw
                        //var name = "picture.jpg";
                        //// 1) open as read-only stream (in MAUI Assets / Resources)
                        //using Stream imgStream = await FileSystem.Current.OpenAppPackageFileAsync(name);
                        //MemoryStream memoryStream = new MemoryStream();
                        //imgStream.CopyTo(memoryStream);
                        //// 2) peek size with Skia (optional)
                        //memoryStream.Position = 0;
                        //imgStream.Position = 0;
                        //if (height == 0)
                        //{
                        //    using var bmp = SKBitmap.Decode(imgStream);
                        //    width = bmp.Width;
                        //    height = bmp.Height;
                        //await GetVisionPrediction(memoryStream);
                        //}
                        ////testing

                        await GetVisionPrediction(stream);
                        await Dispatcher.DispatchAsync(async () =>
                        {
                            canvasView.InvalidateSurface();
                        });

                        loopCount++;
                        if (loopStopwatch.ElapsedMilliseconds >= 1000)
                        {
                            if (debug) Debug.WriteLine($"FPS: {loopCount}");
                            _viewModel.FPS = loopCount;
                            loopCount = 0;
                            loopStopwatch.Restart();
                        }
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine(ex.Message);
                        await Task.Delay(100);
                    }
                }
            });
        }

        int height, width;
        private async Task GetVisionPrediction(Stream stream)
        {
            if (height == 0)
            {
                var image = MLImage.CreateFromStream(stream);
                height = image.Height;
                width = image.Width;
                stream.Position = 0;
            }

            var result = await VisionModelManager.Instance.PoopModel.ProcessImageAsync((stream as MemoryStream).ToArray());
            var box = result.Boxes.FirstOrDefault();

            float scaleResizeX = (float)width / result.InputWidth;
            float scaleResizeY = (float)height / result.InputHeight;

            var res = new PredictionResult
            {
                InputHeight = result.InputHeight,
                InputWidth = result.InputWidth,
                BoundingBoxes = result.Boxes,
                OriginalHeight = height,
                OriginalWidth = width,
                TimeStamp = DateTime.Now,
                InputImage = result.Image,
                //MaskBitmaps = result2.MaskBitmaps,
                //Polygons = result2.Polygons
            };

            //await res.RunSamEncode();
            //await res.RunSamDecode();
            _viewModel.AddPredictionResult(res);
        }

        private void OnCanvasViewPaintSurface(object sender, SKPaintSurfaceEventArgs e)
        {
            try
            {
                if (_viewModel.CurrentPrediction == null || _viewModel.CurrentPrediction.Drawn) return;

                var predictionResult = _viewModel.CurrentPrediction;
                var sourceWidth = predictionResult.OriginalWidth;
                var sourceHeight = predictionResult.OriginalHeight;
                var canvas = e.Surface.Canvas;
                canvas.Clear();

                float scaleX = (float)e.Info.Width / sourceWidth;
                float scaleY = (float)e.Info.Height / sourceHeight;
                float scale = Math.Min(scaleX, scaleY);
                float offsetX = (e.Info.Width - (sourceWidth * scale)) / 2;
                float offsetY = (e.Info.Height - (sourceHeight * scale)) / 2;

                // Draw mask bitmaps
                //if (predictionResult.MaskBitmaps?.Count > 0)
                //{
                //    var maskBmp = predictionResult.MaskBitmaps[0];    // already cropped
                //    canvas.Save();
                //    canvas.Translate(offsetX, offsetY);

                //    float sx = (float)e.Info.Width / maskBmp.Width;
                //    float sy = (float)e.Info.Height / maskBmp.Height;
                //    canvas.Scale(Math.Min(sx, sy));

                //    //Debug.WriteLine($"scale: ({sx}, {sy} ) cam: ({e.Info.Width},{e.Info.Height}) mask:({maskBmp.Width},{maskBmp.Height})");
                //    SKPaint fill = new SKPaint { IsStroke = false, Color = SKColors.Blue.WithAlpha(0x80) };
                //    canvas.DrawBitmap(maskBmp, 0, 0, fill);
                //    canvas.Restore();
                //}

                // Draw bounding boxes
                if (predictionResult.BoundingBoxes.Count > 0)
                {
                    // same letter-box offset
                    canvas.Translate(offsetX, offsetY);

                    // same scale factor you used for boxes
                    canvas.Scale(predictionResult.ScaleResizeX * scale, predictionResult.ScaleResizeY * scale);
                    foreach (var box in predictionResult.BoundingBoxes)
                    {
                        var left = box.Dimensions.X;
                        var top = box.Dimensions.Y;
                        var right = box.Dimensions.X2;
                        var bottom = box.Dimensions.Y2;
                        //var left = offsetX + box.Dimensions.X * predictionResult.ScaleResizeX * scale;
                        //var top = offsetY + box.Dimensions.Y * predictionResult.ScaleResizeY * scale;
                        //var right = offsetX + box.Dimensions.X2 * predictionResult.ScaleResizeX * scale;
                        //var bottom = offsetY + box.Dimensions.Y2 * predictionResult.ScaleResizeY * scale;

                        var paint = new SKPaint
                        {
                            Color = SKColor.Parse(ColorToHex(box.BoxColor)),
                            StrokeWidth = 2,
                            IsStroke = true
                        };

                        var textSize = 26;
#if ANDROID
                        textSize = 40;
#endif
                        var textPaint = new SKPaint
                        {
                            Color = SKColor.Parse(ColorToHex(box.BoxColor)),
                            IsAntialias = true,
                            Style = SKPaintStyle.Fill,
                            TextSize = textSize
                        };

                        canvas.DrawRect(new SKRect(left, top, right, bottom), paint);
                        canvas.DrawText(box.Description, left, top + textSize, textPaint);
                        predictionResult.Drawn = true;
                    }
                }

                //if (predictionResult.Polygons?.Count > 0)
                //{
                //    using var polyStroke = new SKPaint
                //    {
                //        Color = SKColors.Red,
                //        StrokeWidth = 2,
                //        Style = SKPaintStyle.Stroke,
                //        IsAntialias = true
                //    };

                //    using var polyFill = new SKPaint
                //    {
                //        Color = SKColors.Blue.WithAlpha(80),   // translucent fill
                //        Style = SKPaintStyle.Fill,
                //        IsAntialias = true
                //    };

                //    foreach (var flat in predictionResult.Polygons)
                //    {
                //        var pth = BuildPath(flat, offsetX, offsetY, scale);
                //        if (pth == null) continue;

                //        canvas.DrawPath(pth, polyFill);   // interior
                //        canvas.DrawPath(pth, polyStroke); // outline
                //    }
                //}
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
            }
        }
        private static SKPath BuildPath(IReadOnlyList<int> flat,
                                float offsetX, float offsetY, float scale)
        {
            if (flat.Count < 6 || flat.Count % 2 != 0) return null;   // need >=3 pts

            var path = new SKPath();
            path.MoveTo(offsetX + flat[0] * scale,
                        offsetY + flat[1] * scale);

            for (int i = 2; i < flat.Count; i += 2)
                path.LineTo(offsetX + flat[i] * scale,
                            offsetY + flat[i + 1] * scale);

            path.Close();
            return path;
        }

        private static string ColorToHex(System.Drawing.Color c) => $"#{c.R:X2}{c.G:X2}{c.B:X2}";

        // The gear icon (settings) was bound to this handler
        private async void OnSettingsClicked(object sender, EventArgs e)
        {
            // Show the modal that allows user to pick a camera
            // Passing our existing ViewModel so that the selection updates it directly
            await Navigation.PushModalAsync(new CameraSelectionPage(_viewModel));
        }
    }
}
