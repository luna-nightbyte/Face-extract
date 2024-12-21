import cv2
from PIL import Image

from internal.models.face_detector import Face_Detector

def process(image_path: str, OUTPUT_SIZE: tuple[512, 512],model: Face_Detector): 
        
        
        cv2Image = model.get_image(image_path)
        if cv2Image is None:
            return
        
        ih, iw, _ = cv2Image.shape
        print(ih,iw)
        # Detect faces
        import internal.models.face_detector as faceDet
        results = model.detect_face(cv2Image, image_path)
        
        if not results:
            return f"No detections in {image_path}"
        
        if not results.detections:
            return f"No face detected in {image_path}"
        
        
        for i, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            # Convert relative bounding box to absolute coordinates
            x1 = int(bboxC.xmin * iw)
            y1 = int(bboxC.ymin * ih)
            x2 = int((bboxC.xmin + bboxC.width) * iw)
            y2 = int((bboxC.ymin + bboxC.height) * ih)
            # Calculate the center of the face
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            # Determine if resizing is needed
            face_width, face_height = x2 - x1, y2 - y1
            was_resized = face_width > OUTPUT_SIZE[0] or face_height > OUTPUT_SIZE[1]
            if was_resized:
                scale = min(OUTPUT_SIZE[0] / face_width, OUTPUT_SIZE[1] / face_height)
                if scale <= 0:
                    raise ValueError("Scale must be greater than 0")

                cv2Image = cv2.resize(cv2Image, (int(iw * scale), int(ih * scale)))
                center_x = int(center_x * scale)
                center_y = int(center_y * scale)
                ih, iw, _ = cv2Image.shape
            # Define the crop region centered around the face
            half_size = OUTPUT_SIZE[0] // 2
            crop_x1, crop_y1 = max(0, center_x - half_size), max(0, center_y - half_size)
            crop_x2, crop_y2 = min(iw, center_x + half_size), min(ih, center_y + half_size)
            # Ensure the crop region is exactly selected output size with padding if needed
            pad_x1, pad_y1 = max(0, half_size - center_x), max(0, half_size - center_y)
            pad_x2, pad_y2 = max(0, (center_x + half_size) - iw), max(0, (center_y + half_size) - ih)
            cropped_img = cv2Image[crop_y1:crop_y2, crop_x1:crop_x2]
            padded_img = cv2.copyMakeBorder(
                cropped_img,
                pad_y1, pad_y2, pad_x1, pad_x2,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]  # Black padding
            )
            output_image = Image.fromarray(padded_img)
            csv_data = [image_path, f"{center_x},{center_y}", was_resized, f"{x1},{y1},{x2},{y2}"]

            return output_image, csv_data