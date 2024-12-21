# Face extracter

Just a simple face extration script that also saves original image x,y boxes for later use.

This is intended as a project to extract a frame, modify it and place it back in with a different script using the generated csv file.

Use [v1](https://github.com/luna-nightbyte/Face-extract/blob/667b65778aa987a23679a27467661d6d318bd8d0/internal/models/face_detector.py#L106) for best results as v2 is not fully integrated.

```
usage: main.py [-h] --input INPUT [--out OUT] [--size SIZE] [--csv CSV] [--model_selection MODEL_SELECTION] [--confidence CONFIDENCE] [--version VERSION]

Extract faces from images with an extra margin around the face.

options:
  -h, --help            show this help message and exit
  --input INPUT         Folder containing images
  --out OUT             Directory to save the files
  --size SIZE           Extracted image size. Faces exeeding this value will be resized to fit the frame with a decent margin
  --csv CSV             csv filename for later use. Default's to 'data.csv'
  --model_selection MODEL_SELECTION
                        Model selection: 0 or 1. 0 to select a short-range model that works best for faces within 2 meters from the camera, and 1 for a full-range
                        model best for faces within 5 meters. Use as model path for v2 
  --confidence CONFIDENCE
                        Minimum confidence to trigger image saving
  --version VERSION     Model version
```
