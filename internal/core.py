
import argparse
import os
import csv

# Initialize Mediapipe face detection
def run():
    parser = argparse.ArgumentParser(description="Extract faces from images with an extra margin around the face.")
    parser.add_argument("--input", type=str, required=True, help="Folder containing images")
    parser.add_argument("--out", type=str, required=False,  default="output", help="Directory to save the files")
    parser.add_argument("--size", type=int, required=False, default=1024, help="Extracted image size. Faces exeeding this value will be resized to fit the frame with a decent margin")
    parser.add_argument("--csv", type=str, required=False, default="data.csv", help="csv filename for later use. Default's to 'data.csv'")
    parser.add_argument("--model_selection", required=False,default=1, help='Model selection: 0 or 1. 0 to select a short-range model that works best for faces within 2 meters from the camera, and 1 for a full-range model best for faces within 5 meters. Use as model path for v2.')
    parser.add_argument("--confidence", type=float, required=False, default=0.7, help="Minimum confidence to trigger image saving")
    parser.add_argument("--version", type=int, required=False, default=1, help="Model version")

    args = parser.parse_args()
    
    input_dir = args.input
    output_dir = args.out
    csv_file = os.path.join(output_dir, args.csv)
    output_size = (args.size,args.size)
    
    import internal.local as I
    
    internal = I.Internal(
        args.version,
        args.model_selection, 
        input_dir, 
        output_dir, 
        csv_file
        )
    
    if internal.face_detection.version == 2:
        modelName = f"{args.model_selection}"
    else:
        modelName = args.model_selection
        
    internal.face_detection.load_detector(modelName, args.confidence)
    if not internal.face_detection.is_initialyzed():
        print("Error loading model")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    err = internal.csv.write(["filename", "center_x,center_y", "was_resized", "original_bounding_box"])
    if err:
        print(err)
    else:
        err = internal.csv.close()
    if err:
        print(err)
        
    for file in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, file)):
            
            results = internal.face_detection.processor.run(os.path.join(input_dir, file),output_size, internal)
            
            if not results:
                continue
            if type(results) == str:
                print(results)
                continue
            if not results[1]:
                continue
            
            image, data = results[0], results[1]
            
            output_image_path = os.path.join(output_dir,os.path.basename(data[0]))
        
            internal.face_detection.save_image(output_image_path, image)
            
            print(f"Image saved as {output_image_path}")
            # Write csv data for later use
            err = internal.csv.write([data])
            if err:
                print(err)
            else:
                err = internal.csv.close()
            if err:
                print(err)