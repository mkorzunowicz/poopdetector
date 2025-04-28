using Camera.MAUI;
using Microsoft.Maui.Graphics;
using Microsoft.Maui.Storage;
using Microsoft.ML.Data;
using PoopDetector.AI;
using PoopDetector.AI.Vision;
using PoopDetector.ViewModel;
using SkiaSharp;
using SkiaSharp.Views.Maui;
using System.Diagnostics;
using System.IO;
using static Microsoft.Maui.ApplicationModel.Permissions;

namespace PoopDetector.Views;
public partial class CameraPage : ContentPage
{
    private CameraViewModel _viewModel;
    PredictionResult _predictionResult;

    bool playing = false;
    bool debug = true;

    IYolo<IImageInputData> Yolo => AIModelManager.Instance.GetModel();

    public CameraPage()
    {
        InitializeComponent();

        _viewModel = new CameraViewModel();
        BindingContext = _viewModel;
        cameraView.CamerasLoaded += CameraView_CamerasLoaded;
        _viewModel.SelectedCameraChanged += async (camera) => await ChangeCameraAsync(camera);
    }
    private async Task ChangeCameraAsync(CameraInfo newCamera)
    {
        if (playing)
            await cameraView.StopCameraAsync();

        cameraView.Camera = newCamera;

        playing = await cameraView.StartCameraAsync() == CameraResult.Success;
    }
    protected override async void OnAppearing()
    {
        base.OnAppearing();

        StartPredictionLoop();
    }

    protected override void OnDisappearing()
    {
        base.OnDisappearing();
        playing = false;
    }

    //IVision _ultraface;
    //IVision _yoloX;

    //IVision Ultraface => _ultraface ??= new Ultraface();
    //IVision YoloX => _yoloX ??= new YoloX();
    //IVision YoloXNanoPoop => _yoloX ??= new YoloXNanoPoop();
    //IVision YoloXInt8 => _yoloX ??= new YoloXInt8();
    private async void StartPredictionLoop()
    {
        await Task.Factory.StartNew(async () =>
        {
            while (!playing || !AIModelManager.Instance.IsLoaded)
            {
                await Task.Delay(10);
                continue;
            }
            Stopwatch loopStopwatch = Stopwatch.StartNew();
            int loopCount = 0;
            var st = Stopwatch.StartNew();

            while (playing)
            {
                try
                {
                    var stream = cameraView.GetSnapShotStream(Camera.MAUI.ImageFormat.JPEG);
                    //var stream = cameraView.GetSnapShotStream();
                    //var stream = await cameraView.TakePhotoAsync();
                    if (stream == null || !stream.CanRead) continue;


                    if (debug) Debug.WriteLine($"After frame: {st.ElapsedMilliseconds}ms");
                    st.Restart();


                    await GetVisionPrediction(stream);
                    if (debug) Debug.WriteLine($"After Vision detect: {st.ElapsedMilliseconds}ms");
                    st.Restart();

                    //await GetMlPrediction(stream);
                    if (debug) Debug.WriteLine($"After ML detect: {st.ElapsedMilliseconds}ms");
                    st.Restart();


                    canvasView.InvalidateSurface();

                    loopCount++;
                    if (loopStopwatch.ElapsedMilliseconds >= 1000)
                    {
                        if (debug) Debug.WriteLine($"FPS: {loopCount}");
                        _viewModel.FPS = loopCount;
                        loopCount = 0;
                        loopStopwatch.Restart();
                    }
                }
                catch (Exception e)
                {
                    Debug.WriteLine(e.Message);
                }
            }
            st.Stop();
        });
    }

    // 1. We need to make sure the bounding box is of size: 2/4 of the screen, so the camera has to be quite close to the object
    // Or be in a rather high zoom ratio
    // 2. Let's test how high is the the chance that what we see is poop. Let's aim >80%
    // 3. We could also run another model - YOLOv7 to check if what we see isn't something different (but preferably the model should include those)
    // 4. Let's gather 3-5 of such pictures and store them. One stored 640x640 image is about 300kB, so 1,5MB, 100000 pictures = 115GB
    // 5. Calculate the average probability of the 5 pictures
    // 6. Poop overlay on the camera page + some progress circle
    // The progress circle is quite important for slower processors so it's clear, the data is read
    // 7. We could also run some YOLOX L on the server side, just to make sure we see what we should.

    // 8. Pickup confirmation.
    // 8a. We could take a picture of a full poop bag lying next to the poop loction - we can also train a model to recognize those
    // 8b. We could take a picture of the poop location and try to run a comparison of pixels from OpenCV - this could be scetchy.
    // Either way, only when we confirm that both the poop picture as well as confirmation are solid - we save the pic in DB.
    // 9. We could also save some accelerometer data, to make sure the phone didn't move to far from the poop location -
    // like 'hey, where did you go?'
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
        //var result = await YoloXNanoPoop.ProcessImageAsync((stream as MemoryStream).ToArray());
        _predictionResult = _viewModel.CurrentPrediction = new PredictionResult
        {
            InputHeight = result.InputHeight,
            InputWidth = result.InputWidth,
            BoundingBoxes = result.Boxes,
            OriginalHeight = height,
            OriginalWidth = width,
            TimeStamp = DateTime.Now,
            InputImage = result.Image
        };

    }
    private async Task GetMlPrediction(Stream stream)
    {
        var image = MLImage.CreateFromStream(stream);
        var boundingBoxes = await Yolo.DetectObjects(Yolo.GetInputData(image));

        //var result = new PredictionResult { BoundingBoxes = boundingBoxes, InputHeight = image.Height, InputWidth = image.Width, TimeStamp = DateTime.Now, InputImage = image.GetBGRPixels };     
        var result = new PredictionResult { BoundingBoxes = boundingBoxes, OriginalHeight = image.Height, OriginalWidth = image.Width, TimeStamp = DateTime.Now };

        _predictionResult = _viewModel.CurrentPrediction = result;


    }

    private void OnCanvasViewPaintSurface(object sender, SKPaintSurfaceEventArgs e)
    {
        //if (prevFrame == null || _predictionResult == null) return;
        if (_predictionResult == null || _predictionResult.Drawn) return;
        var sourceWidth = _predictionResult.OriginalWidth;
        var sourceHeight = _predictionResult.OriginalHeight;
        var canvas = e.Surface.Canvas;
        canvas.Clear();
        if (_predictionResult.BoundingBoxes.Count == 0) return;
        // Calculate scaling factors
        float scaleX = (float)e.Info.Width / sourceWidth;
        float scaleY = (float)e.Info.Height / sourceHeight;
        float scale = Math.Min(scaleX, scaleY);

        // Calculate offset to center the image
        float offsetX = (e.Info.Width - (sourceWidth * scale)) / 2;
        float offsetY = (e.Info.Height - (sourceHeight * scale)) / 2;
        float scaleResizeX = (float)sourceWidth / (float)VisionModelManager.Instance.PoopModel.InputSize.Width;
        float scaleResizeY = (float)sourceHeight / (float)VisionModelManager.Instance.PoopModel.InputSize.Width;

        foreach (var box in _predictionResult.BoundingBoxes)
        {

            if (box.Label == "face")
            {
                scale = 1;
                scaleResizeX = 1;
                scaleResizeY = 1;
            }
            // Scale the coordinates
            var left = offsetX + box.Dimensions.X * scaleResizeX * scale;
            var top = offsetY + box.Dimensions.Y * scaleResizeY * scale;
            var right = offsetX + (box.Dimensions.X + box.Dimensions.Width) * scaleResizeX * scale;
            var bottom = offsetY + (box.Dimensions.Y + box.Dimensions.Height) * scaleResizeY * scale;

            // Draw rectangles
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
            // Draw the rectangle with scaled coordinates
            canvas.DrawRect(new SKRect(left, top, right, bottom), paint);
            canvas.DrawText(box.Description, left, top + textSize, textPaint); // Adjust position as needed
            _predictionResult.Drawn = true;
        }
    }
    private void CameraView_CamerasLoaded(object sender, EventArgs e)
    {
        if (cameraView.Cameras.Count > 0)
        {
            foreach (var cam in cameraView.Cameras)
                _viewModel.Cameras.Add(cam);
            _viewModel.SelectedCamera = _viewModel.Cameras.First();

        }
    }
    private static String ColorToHex(System.Drawing.Color c) => $"#{c.R:X2}{c.G:X2}{c.B:X2}";
}