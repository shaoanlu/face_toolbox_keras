# face-toolbox-keras

A collection of deep learning models ported to Keras for face analysis 

![](https://github.com/shaoanlu/face-toolbox-keras/raw/master/examples.jpg)

---
## Usage

A jupyter notebook demo on Colab is provided.

### Face detection

The S3FD model for face detection is ported from [1adrianb/face-alignment](https://github.com/1adrianb/face-alignment)

```python
from models.detector import face_detector

im = cv2.imread(...)[..., ::-1]
fd = face_detector.FaceAlignmentDetector()
bboxes = fd.detect_face(im, with_landmarks=False)
```

### Face landmarks detection

The FAN-4 model for landmarks detection is ported from [1adrianb/face-alignment](https://github.com/1adrianb/face-alignment). Lite FAN such as FAN-1 and FAN-2 are also provided.

```python
from models.detector import face_detector

im = cv2.imread(...)[..., ::-1]
fd = face_detector.FaceAlignmentDetector()
bboxes, landmarks = fd.detect_face(im, with_landmarks=True)
```

### Face parsing

The BiSeNet model for face parsing is ported from [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)

```python
from models.parser import face_parser

im = cv2.imread(...)[..., ::-1]
prs = face_parser.FaceParser()
# prs.set_detector(fd)
parsing_map = prs.parse_face(im)
```

### Eye region landmarks detection

The ELG model for eye region landmarks detection is ported from [swook/GazeML](https://github.com/swook/GazeML)

```python
from models.detector.iris_detector import IrisDetector

im = cv2.imread(...)[..., ::-1]
idet = IrisDetector()
idet.set_detector(fd)
eye_lms = idet.detect_iris(im)
```

### Face verification

The InceptionResNetV1 model (model name: 20180402-114759 trained on VGGFace2) for face verification is ported from [davidsandberg/facenet](https://github.com/davidsandberg/facenet)

```python
from models.verifier.face_verifier import FaceVerifier

im1 = cv2.imread(...)[..., ::-1]
im2 = cv2.imread(...)[..., ::-1]
fv = FaceVerifier(classes=512)
# fv.set_detector(fd)
result, distance = fv.verify(im1, im2, threshold=0.5, with_detection=False, return_distance=True)
```
