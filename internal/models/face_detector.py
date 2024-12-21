
import cv2
import mediapipe as mp

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


class Face_Detector:
    def init(self,v: int, name):
        self.detector = None
        self.version = v
        self.mod = None
        self.modelName = None
        if not self.version > 0:
            print("invalid version")
            return
        if self.version == 1:
            self.mod = self.v1()
            self.modelName = name
        if self.version == 2:
            self.mod = self.v2()
            self.modelName = f"{name}"
        return self
    
    def is_initialyzed(self):
        return self.detector is not None  
    
    def name(self):
        return self.modelName  
    
    def load_detector(self, model, conf):
        self.detector = self.mod.load(model,conf)
        
    def detect_face(self, rgb_image: cv2.typing.MatLike, image_path: str ):
        return self.mod.detect(rgb_image,image_path)
    
    def get_image(self, image_path: str ):
        return self.mod.get_image(image_path)
    
    def save_image(self,output_path: str, image ):
        return self.mod.save_image(output_path, image)
 
    def is_valid_model(self, model):
        return Face_Detector() == model
    

    class v2:
        def load(self, model, conf_unused):
            try:
                int(model) + 1
                print("invalid model settings!")
                return None
            except:
                pass

            base_options = python.BaseOptions(model_asset_path=model)
            options = vision.FaceDetectorOptions(base_options=base_options)
            self.detector = vision.FaceDetector.create_from_options(options)
            return self.detector

        def detect(self, rgb_image: cv2.typing.MatLike, image_path: str):
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                print("Image loading failed!")
                return None

            # Convert to RGB for Mediapipe processing
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self.detector.detect(img)
            return convert_detection_results_to_loop_format(results)
        
        def get_image(self, image_path: str ):
            return cv2.imread(image_path)
        
        def save_image(self,output_path: str, image):
            cv2.imwrite(output_path, image)
 

    class v1:
        def load(self,model,conf):
            try:
                int(model)+1
            except:
                return "invalid model settings!"
            self.detector = mp.solutions.face_detection.FaceDetection(
                model_selection=model, 
                min_detection_confidence=conf
                )
            
            return self.detector
        def detect(self, rgb_image: cv2.typing.MatLike, image_path: str): 
            results = self.detector.process(rgb_image)
            if not results.detections:
                return None
            
            return results
        
        def get_image(self, image_path: str ):
            image = cv2.imread(image_path)
            if image is None:
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        def save_image(self, output_path:str, image):
            image.save(output_path)