# YOLOv8 inference using Go

This is a web interface to [YOLOv8 object detection neural network](https://ultralytics.com/yolov8)
implemented on [Go](https://go.dev).

This is a source code for a ["How to create YOLOv8-based object detection web service using Python, Julia, Node.js, JavaScript, Go and Rust"](https://dev.to/andreygermanov/how-to-create-yolov8-based-object-detection-web-service-using-python-julia-nodejs-javascript-go-and-rust-4o8e) tutorial.

## Install

* Clone this repository: `git clone git@github.com:AndreyGermanov/yolov8_onnx_nodejs.git`
* Go to the root of cloned repository
* Install dependencies by running `go get`
* Open the `main.go`, find line `ort.SetSharedLibraryPath(...)` and specify the path to the ONNX runtime library path in it.*

*If you do not have installed `ONNX runtime`, then you can manually download it for
your operating system from [this repository](https://github.com/microsoft/onnxruntime/releases),
extract archive to some folder and then specify a path to a main library path: 
subfolder:

* `lib/libonnxruntime.so` - for Linux
* `lib/libonnxruntime.dylib` - for MacOS
* `lib/onnxruntime.dll` - for Windows

This repository contains the ONNX Runtime library for Linux only.

## Run

Execute:

```
go run main.go
```

It will start a webserver on http://localhost:8080. Use any web browser to open the web interface.

Using the interface you can upload the image to the object detector and see bounding boxes of all objects detected on it.