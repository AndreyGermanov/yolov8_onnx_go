package main

import ort "github.com/yalue/onnxruntime_go"

var (
	UseCoreML  = false
	Blank      []float32
	ModelPath  = "./yolov8m.onnx"
	Yolo8Model ModelSession
)

type ModelSession struct {
	Session *ort.AdvancedSession
	Input   *ort.Tensor[float32]
	Output  *ort.Tensor[float32]
}
