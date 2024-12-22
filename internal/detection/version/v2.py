import cv2
import mediapipe
import os
import numpy

from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class RelativeBoundingBox:
    def __init__(self, x, y, width, height):
        self.xmin = x
        self.ymin = y
        self.width = width
        self.height = height

class LocationData:
    def __init__(self, bounding_box):
        self.relative_bounding_box = bounding_box

class DetectionWrapper:
    def __init__(self, detection):
        bbox = detection.bounding_box
        self.location_data = LocationData(RelativeBoundingBox(bbox.origin_x, bbox.origin_y, bbox.width, bbox.height))

class ResultsWrapper:
    def __init__(self, detections):
        self.detections = [DetectionWrapper(detection) for detection in detections]

def convert_detection_results_to_loop_format(detection_results):
    """
    Converts detection results to a format compatible with the given loop.

    Args:
        detection_results: Original detection results.

    Returns:
        ResultsWrapper: Transformed detection results.
    """
    return ResultsWrapper(detection_results.detections)


class Model:
    def load(self, model, conf_unused):
        try:
            int(model) + 1
            return f"invalid model settings"
        except:
            pass
        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)
        return self.detector
    def detect(self, rgb_image: cv2.typing.MatLike, image_path: str):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return f"Image loading failed!"
        # Convert to RGB for Mediapipe processing
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=image_rgb)
        results = self.detector.detect(img)
        return convert_detection_results_to_loop_format(results)
    
    def get_image(self, image_path: str ):
        image = cv2.imread(image_path)
        if image is None:
            print("Image failed to load!")  # Debugging addition
            return None
        return image
    def save_image(self,output_path: str, image: Image.Image):
        image.save("test1.png")
        output_image = numpy.array(image.convert('RGB'))
        cv2.imwrite(output_path, output_image)

    def get_bbx(self, bboxC,iw , ih):
        return int(bboxC.xmin ), int(bboxC.ymin ), int((bboxC.xmin + bboxC.width) ),int((bboxC.ymin + bboxC.height) )
        return int(bboxC.xmin * iw), int(bboxC.ymin * ih), int((bboxC.xmin + bboxC.width) * iw),int((bboxC.ymin + bboxC.height) * ih)