package main

import (
	"encoding/json"
	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"net/http"
	"os"
	"sort"
)

// Main function that defines
// a web service endpoints a starts
// the web service
func main() {
	server := http.Server{
		Addr: "0.0.0.0:8080",
	}
	http.HandleFunc("/", index)
	http.HandleFunc("/detect", detect)
	server.ListenAndServe()
}

// Site main page handler function.
// Returns Content of index.html file
func index(w http.ResponseWriter, _ *http.Request) {
	file, _ := os.Open("index.html")
	buf, _ := io.ReadAll(file)
	w.Write(buf)
}

// Handler of /detect POST endpoint
// Receives uploaded file with a name "image_file", passes it
// through YOLOv8 object detection network and returns and array
// of bounding boxes.
// Returns a JSON array of objects bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
func detect(w http.ResponseWriter, r *http.Request) {
	r.ParseMultipartForm(0)
	file, _, _ := r.FormFile("image_file")
	boxes := detect_objects_on_image(file)
	buf, _ := json.Marshal(&boxes)
	w.Write(buf)
}

// Function receives an image,
// passes it through YOLOv8 neural network
// and returns an array of detected objects
// and their bounding boxes
// Returns Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
func detect_objects_on_image(buf io.Reader) [][]interface{} {
	input, img_width, img_height := prepare_input(buf)
	output := run_model(input)
	return process_output(output, img_width, img_height)
}

// Function used to convert input image to tensor,
// required as an input to YOLOv8 object detection
// network.
// Returns the input tensor, original image width and height
func prepare_input(buf io.Reader) ([]float32, int64, int64) {
	img, _, _ := image.Decode(buf)
	size := img.Bounds().Size()
	img_width, img_height := int64(size.X), int64(size.Y)
	img = resize.Resize(640, 640, img, resize.Lanczos3)
	red := []float32{}
	green := []float32{}
	blue := []float32{}
	for y := 0; y < 640; y++ {
		for x := 0; x < 640; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			red = append(red, float32(r/257)/255.0)
			green = append(green, float32(g/257)/255.0)
			blue = append(blue, float32(b/257)/255.0)
		}
	}
	input := append(red, green...)
	input = append(input, blue...)
	return input, img_width, img_height
}

// Function used to pass provided input tensor to
// YOLOv8 neural network and return result
// Returns raw output of YOLOv8 network as a single dimension
// array
func run_model(input []float32) []float32 {
	ort.SetSharedLibraryPath("./libonnxruntime.so")
	_ = ort.InitializeEnvironment()

	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, _ := ort.NewTensor(inputShape, input)

	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, _ := ort.NewEmptyTensor[float32](outputShape)

	session, _ := ort.NewSession[float32]("./yolov8m.onnx",
		[]string{"images"}, []string{"output0"},
		[]*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})

	_ = session.Run()
	return outputTensor.GetData()
}

// Function used to convert RAW output from YOLOv8 to an array
// of detected objects. Each object contain the bounding box of
// this object, the type of object and the probability
// Returns array of detected objects in a format [[x1,y1,x2,y2,object_type,probability],..]
func process_output(output []float32, img_width, img_height int64) [][]interface{} {
	boxes := [][]interface{}{}
	for index := 0; index < 8400; index++ {
		class_id, prob := 0, float32(0.0)
		for col := 0; col < 80; col++ {
			if output[8400*(col+4)+index] > prob {
				prob = output[8400*(col+4)+index]
				class_id = col
			}
		}
		if prob < 0.5 {
			continue
		}
		label := yolo_classes[class_id]
		xc := output[index]
		yc := output[8400+index]
		w := output[2*8400+index]
		h := output[3*8400+index]
		x1 := (xc - w/2) / 640 * float32(img_width)
		y1 := (yc - h/2) / 640 * float32(img_height)
		x2 := (xc + w/2) / 640 * float32(img_width)
		y2 := (yc + h/2) / 640 * float32(img_height)
		boxes = append(boxes, []interface{}{float64(x1), float64(y1), float64(x2), float64(y2), label, prob})
	}

	sort.Slice(boxes, func(i, j int) bool {
		return boxes[i][5].(float32) < boxes[j][5].(float32)
	})
	result := [][]interface{}{}
	for len(boxes) > 0 {
		result = append(result, boxes[0])
		tmp := [][]interface{}{}
		for _, box := range boxes {
			if iou(boxes[0], box) < 0.7 {
				tmp = append(tmp, box)
			}
		}
		boxes = tmp
	}
	return result
}

// Function calculates "Intersection-over-union" coefficient for specified two boxes
// https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
// Returns Intersection over union ratio as a float number
func iou(box1, box2 []interface{}) float64 {
	return intersect(box1, box2) / union(box1, box2)
}

// Function calculates union area of two boxes
// Returns Area of the boxes union as a float number
func union(box1, box2 []interface{}) float64 {
	box1_x1, box1_y1, box1_x2, box1_y2 := box1[0].(float64), box1[1].(float64), box1[2].(float64), box1[3].(float64)
	box2_x1, box2_y1, box2_x2, box2_y2 := box2[0].(float64), box2[1].(float64), box2[2].(float64), box2[3].(float64)
	box1_area := (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
	box2_area := (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
	return box1_area + box2_area - intersect(box1, box2)
}

// Function calculates intersection area of two boxes
// Returns Area of intersection of the boxes as a float number
func intersect(box1, box2 []interface{}) float64 {
	box1_x1, box1_y1, box1_x2, box1_y2 := box1[0].(float64), box1[1].(float64), box1[2].(float64), box1[3].(float64)
	box2_x1, box2_y1, box2_x2, box2_y2 := box2[0].(float64), box2[1].(float64), box2[2].(float64), box2[3].(float64)
	x1 := math.Max(box1_x1, box2_x1)
	y1 := math.Max(box1_y1, box2_y1)
	x2 := math.Min(box1_x2, box2_x2)
	y2 := math.Min(box1_y2, box2_y2)
	return (x2 - x1) * (y2 - y1)
}

// Array of YOLOv8 class labels
var yolo_classes = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
	"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
	"suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
	"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
	"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}
