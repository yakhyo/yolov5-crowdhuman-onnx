import cv2
import onnxruntime
import numpy as np

import torch
import torchvision

from typing import Tuple, List


class YOLOv5:
    def __init__(self, model_path: str, conf_thres: float = 0.25, iou_thres: float = 0.45, max_det: int = 300, nms_mode: str = 'torchvision') -> None:
        """YOLOv5 class initialization

        Args:
            model_path (str): Path to .onnx file
            conf_thres (float, optional): Confidence threshold. Defaults to 0.25.
            iou_thres (float, optional): IOU threshold. Defaults to 0.45.
            max_det (int, optional): Maximum number of detections. Defaults to 300.
            nms_mode (str, optional): NMS calculation method. Defaults to `torchvision`
        """
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.max_det = max_det
        self.nms_mode = nms_mode

        # Initialize model
        self._initialize_model(model_path=model_path)

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the model on the given image and return predictions.

        Args:
            image (np.ndarray): Input image.

        Returns:
            Tuple: boxes, confidence scores, class indexes
        """
        if not isinstance(image, np.ndarray) or len(image.shape) != 3:
            raise ValueError("Input image must be a numpy array with 3 dimensions (H, W, C).")

        outputs = self.inference(image)
        predictions = self.postprocess(outputs)
        return predictions

    def inference(self, image: np.ndarray) -> List[np.ndarray]:
        """Run inference on the given image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            List[np.ndarray]: Model outputs.
        """
        input_tensor = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def _initialize_model(self, model_path: str) -> None:
        """Initialize the model from the given path.

        Args:
            model_path (str): Path to .onnx model.
        """
        try:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            # Get model info
            self.output_names = [x.name for x in self.session.get_outputs()]
            self.input_names = [x.name for x in self.session.get_inputs()]

            # Get model metadata
            metadata = self.session.get_modelmeta().custom_metadata_map
            self.stride = int(metadata.get("stride", 32))  # Default stride value
            self.names = eval(metadata.get("names", "{}"))  # Default to empty dict
        except Exception as e:
            print(f"Failed to load the model: {e}")
            raise

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocessing

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: HWC -> CHW, BGR to RGB, Normalize and Add batch dimension.
        """
        image = image.transpose(2, 0, 1)  # Convert from HWC -> CHW
        image = image[::-1]  # Convert BGR to RGB
        image = np.ascontiguousarray(image)
        image = image.astype(np.float32) / 255.0  # Normalize the input
        image_tensor = image[np.newaxis, ...]  # Add batch dimension

        return image_tensor

    def postprocess(self, prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Post processing

        Args:
            prediction (np.ndarray): Model raw output (shape: [1, 25200, 85])

        Returns:
            Tuple: boxes, confidence scores, class indexes
        """
        # Squeeze the output to match the expected shape
        outputs = np.squeeze(prediction[0])

        # Extract boxes, scores, and classes
        boxes = outputs[:, :4]  # x1, y1, x2, y2
        scores = outputs[:, 4]  # confidence scores
        classes = outputs[:, 5:]  # class probabilities

        boxes = self.xywh2xyxy(boxes)

        # Apply confidence threshold
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        # Get class with highest probability for each detection
        class_ids = np.argmax(classes, axis=1)[:self.max_det]

        # Apply NMS
        if self.nms_mode == "torchvision":
            # better performance
            indices = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), self.iou_threshold).numpy()
        else:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)

        boxes, scores, class_ids = boxes[indices], scores[indices], class_ids[indices]

        return boxes, scores, class_ids

    def xywh2xyxy(self, x: np.ndarray) -> np.ndarray:
        """xywh -> xyxy

        Args:
            x (np.ndarray): [x, y, w, h]

        Returns:
            np.ndarray: [x1, y1, x2, y2]
        """
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y
