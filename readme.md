# Poop detector

The .NET 9.0 MAUI solution which allows running live inference with ONNX models using camera frames.
I trained a Nano YOLOX model to make it as fast as possible, since I expected to run a live camera feed. 
On Samsung S9 I'm getting 4-5 frames per second.

Segmentation is executed after the frame is frozen using Mobile SAM. It points either in the middle of a bounding box of the detected poop (even if it's false positive), or in the middle of the frame.
The encoding takes about a second on the said S9, decoding is practically instant.