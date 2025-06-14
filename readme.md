# Poop detector

The .NET 9.0 MAUI solution which allows running live inference with ONNX models using camera frames.
I trained a Nano YOLOX model to make it as fast as possible, since I expected to run a live camera feed.

The AI was trained on [Erotemic's Shitspotter database](https://github.com/Erotemic/shitspotter).

# Usage

There are 3 tabs - Camera, Gallery and Models.

## Camera
Camera is the default tab - and the YoloXPoop model starts by default. Inference is executed as soon as the model loads.

The red number in the upper left corner indicates FPS.

It recognizes poop and freezes a frame automatically if an object is of proper size on the screen. It gives on screen information to move closer or further away.
So far the models are far from perfect and there's a lot of false positives, espacially on dark objects, they don't even need to resamble poop. The YoloX implementation is especially fond of rectangular and even shaped objects. Not sure why.

A picture can be forced by clicking the shutter button.

The button on the right is Camera selection modal view. It also allows to turn of SAM.

### Segmentation

By default, on the frozen frame segmentation is executed. If there's an object it points selects the middle of the bounding box as the decode target. You can clear the mask by pressing the broom button in the middle. Pressing on the picture will run the decode by adding more points for segmentation.

Pressing the green thumbs up button will save the picture with the mask to the Gallery.

Red thumbs down cancels and returns to the Camera view, where inference continues.

## Gallery

Is the place where saved pictures along with their masks is saved to.

Those are 2 separate files.

There's no option to download those files yet.

## Models

### Preconfigured
It's where you can choose models.
YoloXNanoPoop and YoloXNano (which is there for sanity check) come with the app. So are SAM models btw.
Yolov9Shitspotter is about 200MB and is first downloaded from the hugginface servers.

### Downloadable
You can downlod other models for YoloX and Yolov9, but there's only one BoundingBox color defined for those.

Once a model is downloaded it is cached on the device - can be deleted by removing the app or clearing its cache.
If a downloaded model's name matches the previously downloaded model it will use the cached one. So if you want to test a new model - make sure the name is different or clear the cache/app.

# Benchmarks

There are 2 crucial time factors: Fetching a frame and inference. Resolution of the camera frame might impact the frame stream access time, as well as inference (as part of it is picture resize).

## iOS

Tested on iPhone 15.
CoreML is doing great.

SAM encode takes about 2500ms. Decode 250ms.
YoloX Poop inference takes 50ms.
Yolov9 Shitspotter takes about 400ms.

## Android

Tested on Samsung Galaxy S10.
NNAPI is apparently deprecated and the worst part is, it slows inference down, so CPU is actually performing better. Even though NNAPI shows certain Operations are supported, it doesn't necessarily mean the underlying drivers support them and it could fallback to CPU. I have no idea why would it slow down though. Currently it's set to use CPU on Androids. Further testing/in App switch could make sense.
Might be we need to either optimize or quantify the models properly. But my initial benchmark shows none of those options helped.

SAM encode takes about 4000ms. Decode 300ms.
YoloX Poop inference takes 200ms.
Yolov9 Shitspotter takes about 30000ms.

## Windows

It runs on CPU. I tried using the CUDA implementation but I failed.

# TODOs and bugs

1. Polygons could be done as concave rather than convex, as it now marks oval, rather than starlike outlines. That could increase complexity even further.
They are currently turned off as the runtime tends to run out of memory on mobile devices if the selected object is to big/complex. 
Those are particularly nasty as they just crash the whole app, since it's a native runtime error.

2. Printing masks or resizing the app window on Windows sometimes crashes the app. Apparently some Skia implementation issue.

3. The pictures from Gallery cannot be downloaded.

4. Downloading external models might sometimes fail on Android. Fortunately if restarted the download resumes instead of starting over.

5. Downloading and loading models show the same progress bar.

6. Saving masked pictures sometimes blows up on iOS. Skia native error again.
