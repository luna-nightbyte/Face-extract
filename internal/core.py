
import argparse
import os
import csv
import cv2


import internal.detection as detection
import internal.models.face_detector as faceDet

AI = None
# Initialize Mediapipe face detection
def run():
    parser = argparse.ArgumentParser(description="Extract faces from images with an extra margin around the face.")
    parser.add_argument("--input", type=str, required=True, help="Folder containing images")
    parser.add_argument("--out", type=str, required=False,  default="output", help="Directory to save the files")
    parser.add_argument("--size", type=int, required=False, default=1024, help="Extracted image size. Faces exeeding this value will be resized to fit the frame with a decent margin")
    parser.add_argument("--csv", type=str, required=False, default="data.csv", help="csv filename for later use. Default's to 'data.csv'")
    parser.add_argument("--model_selection", required=False,default=1, help="Model selection. See: ")
    parser.add_argument("--confidence", type=float, required=False, default=0.7, help="Minimum confidence to trigger image saving")
    parser.add_argument("--version", type=int, required=False, default=1, help="Model version")

    args = parser.parse_args()
    
    input_dir = args.input
    output_dir = args.out
    csv_file = os.path.join(output_dir, args.csv)
    output_size = (args.size,args.size)
    conf = args.confidence
    
    AI = faceDet.Face_Detector().init(args.version, args.model_selection)
    
    AI.load(AI.name(), conf)
    if not AI.up():
        print("Error loading model")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "center_x,center_y", "was_resized", "original_bounding_box"])
        csvfile.close()
        
    for file in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, file)):
            
            results = detection.process(os.path.join(input_dir, file),output_size,AI)
            if not results:
                continue
            if type(results) == str:
                print(results)
                continue
            if not results[1]:
                continue
            
            image, data = results[0], results[1]
            
            output_image_path = os.path.join(output_dir,os.path.basename(data[0]))
            
            image.save(output_image_path)
            print(f" output image saved as {output_image_path}")
            # Write csv data for later use
            with open(csv_file, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows([data])
            csvfile.close()