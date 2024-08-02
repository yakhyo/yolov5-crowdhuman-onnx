import os
import cv2
import logging
import argparse
import numpy as np
from pathlib import Path

from typing import List, Tuple

from models import YOLOv5
from utils.general import check_img_size, scale_boxes, draw_detections, colors, increment_path, LoadMedia


def run_object_detection(
    weights: str,
    source: str,
    img_size: List[int],
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    save: bool,
    view: bool,
    project: str,
    name: str
):
    if save:
        save_dir = increment_path(Path(project) / name)
        save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLOv5(weights, conf_thres, iou_thres, max_det)
    img_size = check_img_size(img_size, s=model.stride)
    dataset = LoadMedia(source, img_size=img_size)

    # For writing video and webcam
    vid_writer = None
    if save and dataset.type in ["video", "webcam"]:
        cap = dataset.cap
        save_path = str(save_dir / os.path.basename(source))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for resized_image, original_image, status in dataset:
        # Model Inference
        boxes, scores, class_ids = model(resized_image)

        # Scale bounding boxes to original image size
        boxes = scale_boxes(resized_image.shape, boxes, original_image.shape).round()

        # Draw bunding boxes
        for box, score, class_id in zip(boxes, scores, class_ids):
            draw_detections(original_image, box, score, model.names[int(class_id)], colors(int(class_id)))

        # Print results
        for c in np.unique(class_ids):
            n = (class_ids == c).sum()  # detections per class
            status += f"{n} {model.names[int(c)]}{'s' * (n > 1)}, "  # add to string

        if view:
            # Display the image with detections
            cv2.imshow('Webcam Inference', original_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

        print(status)

        if save:
            if dataset.type == "image":
                save_path = str(save_dir / f"frame_{dataset.frame:04d}.jpg")
                cv2.imwrite(save_path, original_image)
            elif dataset.type in ["video", "webcam"]:
                vid_writer.write(original_image)

    if save and vid_writer is not None:
        vid_writer.release()

    if save:
        print(f"Results saved to {save_dir}")

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/yolov5m.onnx", help="model path")
    parser.add_argument("--source", type=str, default="0", help="Path to video/image/webcam")
    parser.add_argument("--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.45, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--save", action="store_true", help="Save detected images")
    parser.add_argument("--view", action="store_true", help="View inferenced images")
    parser.add_argument("--project", default="runs", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    args = parser.parse_args()
    args.img_size = args.img_size * 2 if len(args.img_size) == 1 else args.img_size
    return args


def download_weights(weights):
    pass


def main():
    params = parse_args()
    run_object_detection(
        weights=params.weights,
        source=params.source,
        img_size=params.img_size,
        conf_thres=params.conf_thres,
        iou_thres=params.iou_thres,
        max_det=params.max_det,
        save=params.save,
        view=params.view,
        project=params.project,
        name=params.name
    )


if __name__ == "__main__":
    main()
