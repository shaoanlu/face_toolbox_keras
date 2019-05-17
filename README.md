# face-toolbox-keras

A collection of deep learning frameworks for face analysis ported to Keras. 

![](https://github.com/shaoanlu/face-toolbox-keras/raw/master/examples.jpg)

---
## Usage

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaoanlu/face-toolbox-keras/blob/master/demo.ipynb)

### 1. Face detection
```python
from models.detector import face_detector

im = cv2.imread(PATH_TO_IMAGE)[..., ::-1]
fd = face_detector.FaceAlignmentDetector()
bboxes = fd.detect_face(im, with_landmarks=False)
```

### 2. Face landmarks detection

The default model is FAN-4. Lite models of FAN-1 and FAN-2 are also provided.

```python
from models.detector import face_detector

im = cv2.imread(PATH_TO_IMAGE)[..., ::-1]
fd = face_detector.FaceAlignmentDetector()
bboxes, landmarks = fd.detect_face(im, with_landmarks=True)
```

### 3. Face parsing
```python
from models.parser import face_parser

im = cv2.imread(PATH_TO_IMAGE)[..., ::-1]
prs = face_parser.FaceParser()
# prs.set_detector(fd) # fd = face_detector.FaceAlignmentDetector()
parsing_map = prs.parse_face(im)
```

### 4. Eye region landmarks detection

Faster face detection using MTCNN can be found in [this](https://github.com/shaoanlu/GazeML-keras) repo.

```python
from models.detector impoort iris_detector

im = cv2.imread(PATH_TO_IMAGE)[..., ::-1]
idet = iris_detector.IrisDetector()
idet.set_detector(fd) # fd = face_detector.FaceAlignmentDetector()
eye_lms = idet.detect_iris(im)
```

### 5. Face verification
```python
from models.verifier import face_verifier

im1 = cv2.imread(PATH_TO_IMAGE1)[..., ::-1]
im2 = cv2.imread(PATH_TO_IMAGE2)[..., ::-1]
fv = face_verifier.FaceVerifier(classes=512)
# fv.set_detector(fd) # fd = face_detector.FaceAlignmentDetector()
result, distance = fv.verify(im1, im2, threshold=0.5, with_detection=False, return_distance=True)
```

## Requirements
- Keras 2.2.4
- TensorFlow 1.12.0 or 1.13.1

## Acknowledgments
- The S3FD model for face detection is ported from [1adrianb/face-alignment](https://github.com/1adrianb/face-alignment).
- The FAN-4 model for landmarks detection is ported from [1adrianb/face-alignment](https://github.com/1adrianb/face-alignment).
- The BiSeNet model for face parsing is ported from [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch).
- The ELG model for eye region landmarks detection is ported from [swook/GazeML](https://github.com/swook/GazeML). 
- The InceptionResNetV1 model (model name: 20180402-114759 trained on VGGFace2) for face verification is ported from [davidsandberg/facenet](https://github.com/davidsandberg/facenet).
