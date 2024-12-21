# Face extracter

Just a simple face extration script that also saves original image x,y boxes for later use.

This is intended as a project to extract a frame, modify it and place it back in with a different script using the generated csv file.

Use [v1](https://github.com/luna-nightbyte/Face-extract/blob/667b65778aa987a23679a27467661d6d318bd8d0/internal/models/face_detector.py#L106) for best results as v2 is not fully integrated.

```
usage: main.py [-h] --input INPUT [--out OUT] [--size SIZE] [--csv CSV] [--model_selection MODEL_SELECTION]
               [--confidence CONFIDENCE] [--version VERSION]
```
