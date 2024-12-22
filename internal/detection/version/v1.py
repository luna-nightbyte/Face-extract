import mediapipe
import cv2
class Model:
    def load(self,model,conf):
        try:
            int(model)+1
        except:
            return "invalid model settings!"
        self.detector = mediapipe.solutions.face_detection.FaceDetection(
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