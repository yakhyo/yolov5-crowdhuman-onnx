import os
import cv2
import time
import math
import numpy as np
from pathlib import Path


class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance


def check_img_size(imgsz, s=32):
    """Adjusts image size to be divisible by stride `s`, supports int or list/tuple input, returns adjusted size."""

    def make_divisible(x, divisor): return math.ceil(x / divisor) * divisor

    imgsz = list(imgsz)  # convert to list if tuple
    new_size = [make_divisible(x, s) for x in imgsz]

    return new_size


def increment_path(path,  sep="", mkdir=False):
    """
    Generates an incremented file or directory path if it exists, with optional mkdir; args: path, sep="", mkdir=False.

    Example: runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc
    """
    path = Path(path)  # os-agnostic
    if path.exists():
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True)  # make directory

    return path


def letterbox(image, target_shape=(640, 640), color=(114, 114, 114)):
    """Resizes and pads image to target_shape, returns resized image"""
    height, width = image.shape[:2]

    # Calculate scale and new size
    scale = min(target_shape[0] / height, target_shape[1] / width)
    new_size = (int(width * scale), int(height * scale))

    # Resize the image
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    dw, dh = (target_shape[1] - new_size[0]) / 2, (target_shape[0] - new_size[1]) / 2
    top, bottom = int(dh), int(target_shape[0] - new_size[1] - int(dh))
    left, right = int(dw), int(target_shape[1] - new_size[0] - int(dw))

    # Apply padding
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, scale, (dw, dh)


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescales (xyxy) bounding boxes from img1_shape to img0_shape, optionally using provided `ratio_pad`."""
    scale = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # scale  = old / new
    dw, dh = (img1_shape[1] - img0_shape[1] * scale) / 2, (img1_shape[0] - img0_shape[0] * scale) / 2  # wh padding

    boxes[..., [0, 2]] -= dw  # x padding
    boxes[..., [1, 3]] -= dh  # y padding
    boxes[..., :4] /= scale

    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes, shape):
    """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def get_txt_color(color=(128, 128, 128), txt_color=(255, 255, 255)):
    dark_colors = {
        (235, 219, 11),
        (243, 243, 243),
        (183, 223, 0),
        (221, 111, 255),
        (0, 237, 204),
        (68, 243, 0),
        (255, 255, 0),
        (179, 255, 1),
        (11, 255, 162),
    }
    light_colors = {
        (255, 42, 4),
        (79, 68, 255),
        (255, 0, 189),
        (255, 180, 0),
        (186, 0, 221),
        (0, 192, 38),
        (255, 36, 125),
        (104, 0, 123),
        (108, 27, 255),
        (47, 109, 252),
        (104, 31, 17),
    }

    """Assign text color based on background color."""
    if color in dark_colors:
        return 104, 31, 17
    elif color in light_colors:
        return 255, 255, 255
    else:
        return txt_color


def draw_detections(image, box, score, class_name, color):
    x1, y1, x2, y2 = map(int, box)
    label = f"{class_name} {score:.2f}"

    # Draw the bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Calculate text size and scale
    font_size = min(image.shape[:2]) * 0.0006

    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)

    # Create filled rectangle for text background
    cv2.rectangle(image, (x1, y1 - int(1.3 * text_height)), (x1 + text_width, y1), color, -1)

    # Put text on the image
    cv2.putText(
        image,
        label,
        (x1, y1 - int(0.3 * text_height)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        get_txt_color(color),
        1,
        lineType=cv2.LINE_AA
    )


VID_FORMATS = ['mp4', 'avi', 'mov']
IMG_FORMATS = ['jpg', 'jpeg', 'png']


class LoadMedia:
    """YOLOv5 media dataloader for a single image, video or webcam."""

    def __init__(self, path, img_size=(640, 640)):
        """Initializes YOLOv5 loader for a single image or video."""
        self.img_size = img_size
        self.frames = 0

        if path.isdigit() or path == '0':
            self.type = 'webcam'
            self.cap = cv2.VideoCapture(int(path))
        else:
            file_extension = path.split(".")[-1].lower()
            if file_extension in VID_FORMATS:
                self.type = 'video'
                self.cap = cv2.VideoCapture(path)
                self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            elif file_extension in IMG_FORMATS:
                self.type = 'image'
                self.image = cv2.imread(path)
            else:
                raise ValueError(f"Unsupported format: {path}")

        if self.type == "webcam":
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.path = path

    def __iter__(self):
        """Initializes iterator by resetting count and returns the iterator object itself."""
        self.frame = 0  # Resetting the frame count
        return self

    def __next__(self):
        """Advances to the next frame in the video or returns the image, raising StopIteration if at the end."""
        if self.type in ["video", "webcam"]:
            self.cap.grab()
            ret, original_frame = self.cap.retrieve()
            if not ret:
                self.cap.release()
                raise StopIteration
            self.frame += 1
            if self.type == "webcam":
                status = f"{self.type} (frame {self.frame}) Webcam: [{self.path}]: "
            else:
                status = f"{self.type} (frame {self.frame}/{self.frames}) {self.path}: "
        else:
            original_frame = self.image
            if self.frame > 0:
                raise StopIteration
            self.frame += 1
            status = f"{self.type} {self.path}: "

        resized_frame = letterbox(original_frame, self.img_size)[0]  # resize

        return resized_frame, original_frame, status

    def __len__(self):
        """Returns the number of frames in the video or 1 for an image."""
        return self.frames if self.type == 'video' else 1
