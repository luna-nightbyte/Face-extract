
import cv2
import os
import mediapipe as mp

from PIL import Image

from internal.detection.version import v1, v2    

class Core:
    def __init__(self,v: int, name):
        self.detector = None
        self.version = v
        self.mod = None
        self.modelName = None
        if not self.version > 0:
            print("invalid version")
            return
        if self.version == 1:
            self.model = v1.Model()
            self.modelName = name
        if self.version == 2:
            self.model = v2.Model()
            self.modelName = f"{name}"
        
    
    def is_initialyzed(self):
        return self.detector is not None  
    
    def name(self):
        return self.modelName  
    
    def load_detector(self, model, conf):
        self.detector = self.model.load(model,conf)
        
    def detect_face(self, rgb_image: cv2.typing.MatLike, image_path: str ):
        return self.model.detect(rgb_image,image_path)
    
    def get_image(self, image_path: str ):
        return self.model.get_image(image_path)
    
    def save_image(self,output_path: str, image ):
        return self.model.save_image(output_path, image)
 
    def bbox(self,bboxC,iw , ih ):
        return self.model.get_bbx(bboxC=bboxC,iw=iw,ih=ih)
 
    def is_valid_model(self, model):
        return Core() == model
    
    def run_detection_loop(self, image_path: str, OUTPUT_SIZE: tuple[int, int]):
        print(f"Running detection loop for image: {image_path} with output size: {OUTPUT_SIZE}")

        # Load the image
        cv2Image = self.get_image(image_path)
        if cv2Image is None:
            print(f"Failed to load image from path: {image_path}")
            return f"Failed to load image from {image_path}"

        ih, iw, _ = cv2Image.shape
        print(f"Image loaded with dimensions: {iw}x{ih}")

        # Detect faces
        results = self.detect_face(cv2Image, image_path)
        if not results:
            print(f"Detection results are empty for image: {image_path}")
            return f"No detections in {image_path}"

        if not results.detections:
            print(f"No faces detected in image: {image_path}")
            return f"No face detected in {image_path}"

        for i, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box

            # Convert bounding box to absolute coordinates
            x1, x2, y1, y2 = self.bbox(bboxC, iw, ih)
            
            if x1 >= iw or y1 >= ih or x2 <= 0 or y2 <= 0:
                print(f"Invalid bounding box after scaling: [x1={x1}, y1={y1}, x2={x2}, y2={y2}]")
                return f"Invalid bounding box in {image_path}"
            print(f"NEW Detection {i}: Bounding box [x1={x1}, y1={y1}, x2={x2}, y2={y2}]")
            # Calculate the center of the face
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            print(f"Face center: ({center_x}, {center_y})")

            # Resize if necessary
            face_width, face_height = x2 - x1, y2 - y1
            was_resized = face_width > OUTPUT_SIZE[0] or face_height > OUTPUT_SIZE[1]
            print(f"Face dimensions (width={face_width}, height={face_height}), resizing needed: {was_resized}")

            if was_resized:
                scale = min(OUTPUT_SIZE[0] / face_width, OUTPUT_SIZE[1] / face_height)
                print(f"Scaling factor: {scale}")
                if scale <= 0:
                    raise ValueError("Invalid scale factor")

                cv2Image = cv2.resize(cv2Image, (int(iw * scale), int(ih * scale)))
                center_x = int(center_x * scale)
                center_y = int(center_y * scale)
                ih, iw, _ = cv2Image.shape
                print(f"Image resized to: {iw}x{ih}")

            # Define crop region
            half_size = OUTPUT_SIZE[0] // 2
            crop_x1, crop_y1 = max(0, center_x - half_size), max(0, center_y - half_size)
            crop_x2, crop_y2 = min(iw, center_x + half_size), min(ih, center_y + half_size)
            print(f"Crop region: [x1={crop_x1}, y1={crop_y1}, x2={crop_x2}, y2={crop_y2}]")

            # Add padding if necessary
            pad_x1, pad_y1 = max(0, half_size - center_x), max(0, half_size - center_y)
            pad_x2, pad_y2 = max(0, (center_x + half_size) - iw), max(0, (center_y + half_size) - ih)
            print(f"Padding: [top={pad_y1}, bottom={pad_y2}, left={pad_x1}, right={pad_x2}]")

            cropped_img = cv2Image[crop_y1:crop_y2, crop_x1:crop_x2]
            if cropped_img.size == 0:
                print("Error: Cropping resulted in an empty image!")
                return f"Empty crop region in {image_path}"

            padded_img = cv2.copyMakeBorder(
                cropped_img,
                pad_y1, pad_y2, pad_x1, pad_x2,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]  # Black padding
            )
            print(f"Padded image shape: {padded_img.shape}")

            # Convert to PIL image
            output_image = Image.fromarray(padded_img)
            csv_data = [image_path, f"{center_x},{center_y}", was_resized, f"{x1},{y1},{x2},{y2}"]

            # Debugging output
            print(f"Output image generated for detection {i}")
            print(f"CSV data: {csv_data}")

            return output_image, csv_data
