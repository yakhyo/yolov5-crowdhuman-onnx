# YOLOv5 CrowdHuman ONNX Runtime

![Downloads](https://img.shields.io/github/downloads/yakhyo/yolov5-crowdhuman-onnx/total) [![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yakhyo/yolov5-crowdhuman-onnx)

<video controls autoplay loop src="https://github.com/user-attachments/assets/ade0f186-4dfe-4974-b1e0-a803a2fbd36a" muted="false" width="100%"></video>


Video by Coverr from Pexels: https://www.pexels.com/video/black-and-white-video-of-people-853889/

This repository contains code and instructions for performing object detection using the YOLOv5 model with the CrowdHuman dataset, utilizing ONNX Runtime for inference.

## Features

- Inference using ONNX Runtime with GPU (tested on Ubuntu).
- Easy-to-use Python scripts for inference.
- Supports multiple input formats: image, video, or webcam.

## Installation

#### Clone the Repository

```bash
git clone https://github.com/yakhyo/yolov5-crowdhuman-onnx.git
cd yolov5-crowdhuman-onnx
```

#### Install Required Packages

```bash
pip install -r requirements.txt
```

## Usage

Before running inference, you need to download weights of the YOLOv5m model trained on CrowdHuman dataset in ONNX format.

#### Download weights (Linux)

```bash
sh download.sh
```

#### Download weights from the following links

**Note:** The weights are saved in FP32.

| Model Name | ONNX Model Link                                                                                           | Number of Parameters | Model Size |
| ---------- | --------------------------------------------------------------------------------------------------------- | -------------------- | ---------- |
| YOLOv5m    | [crowdhuman.onnx](https://github.com/yakhyo/yolov5-crowdhuman-onnx/releases/download/v0.0.1/crowdhuman.onnx) | 21.2M                | 84 MB      |

<br>

> If you have custom weights, you can convert your weights to ONNX format. Follow the instructions in the [YOLOv5 repository](https://github.com/ultralytics/yolov5) to convert your model. You can use the converted ONNX model with this repository.

#### Inference

```bash
python main.py --weights weights/crowdhuman.onnx --source assets/vid_input.mp4 # video
                                                 --source 0 --view # webcam and display
                                                 --source assets/img_input.jpg # image
```

- To save results add the `--save` argument and results will be saved under the `runs` folder
- To display video add the `--view` argument

**Command Line Arguments**

```
usage: main.py [-h] [--weights WEIGHTS] [--source SOURCE] [--img-size IMG_SIZE [IMG_SIZE ...]] [--conf-thres CONF_THRES] [--iou-thres IOU_THRES]
               [--max-det MAX_DET] [--save] [--view] [--project PROJECT] [--name NAME]

options:
  -h, --help            show this help message and exit
  --weights WEIGHTS     model path
  --source SOURCE       Path to video/image/webcam
  --img-size IMG_SIZE [IMG_SIZE ...]
                        inference size h,w
  --conf-thres CONF_THRES
                        confidence threshold
  --iou-thres IOU_THRES
                        NMS IoU threshold
  --max-det MAX_DET     maximum detections per image
  --save                Save detected images
  --view                View inferenced images
  --project PROJECT     save results to project/name
  --name NAME           save results to project/name
```

## Reference

1. https://github.com/ultralytics/yolov5
2. Thanks for the model weight to [SibiAkkash](https://github.com/SibiAkkash/yolov5-crowdhuman)
