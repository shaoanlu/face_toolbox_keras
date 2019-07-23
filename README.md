# face-toolbox-keras

A collection of deep learning frameworks ported to Keras for face detection, face segmentation, face parsing, iris detection, and face verification. 

![](https://github.com/shaoanlu/face-toolbox-keras/raw/master/examples.jpg)

## Descriptions

This repository contains deep learning frameworks that we collected and ported to Keras. We wrapped those models into separate modules that aim to provide their functionality to users within 3 lines of code.

- **Face detection:** The S3FD model is ported from [1adrianb/face-alignment](https://github.com/1adrianb/face-alignment).
- **Face landmarks detection:** The 2DFAN-4, 2DFAN-2, and 2DFAN-1 models are ported from [1adrianb/face-alignment](https://github.com/1adrianb/face-alignment).
- **Face parsing:** The BiSeNet model is ported from [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch).
- **Eye region landmarks detection:** The ELG model is ported from [swook/GazeML](https://github.com/swook/GazeML). 
- **Face verification:** The InceptionResNetV1 model (model name: 20180402-114759) is ported from [davidsandberg/facenet](https://github.com/davidsandberg/facenet).
- **Face verification:** The LResNet100E-IR model is ported from [deepinsight/insightface](https://github.com/deepinsight/insightface).
- **Gender and age estimation:** The MobileNet model is ported from [deepinsight/insightface](https://github.com/deepinsight/insightface).

###### *Each module follows the license of their source repo.

## Usage

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaoanlu/face-toolbox-keras/blob/master/demo.ipynb)
 
 This demo requires a GPU instance.

### 1. Face detection
```python
from models.detector import face_detector

im = cv2.imread(PATH_TO_IMAGE)[..., ::-1]
fd = face_detector.FaceAlignmentDetector()
bboxes = fd.detect_face(im, with_landmarks=False)
```

### 2. Face landmarks detection

The default model is 2DFAN-4. Lite models of 2DFAN-1 and 2DFAN-2 are also provided.

| GPU | 2DFAN-1 | 2DFAN-2 | 2DFAN-4 |
|:---:|:-------:|:-------:|:-------:|
| K80 | 74.3ms  | 92.2ms  | 133ms   |

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
fp = face_parser.FaceParser()
# fp.set_detector(fd) # fd = face_detector.FaceAlignmentDetector()
parsing_map = fp.parse_face(im, bounding_box=None, with_detection=False)
```

### 4. Eye region landmarks detection

Faster face detection using MTCNN can be found in [this](https://github.com/shaoanlu/GazeML-keras) repo.

```python
from models.detector import iris_detector

im = cv2.imread(PATH_TO_IMAGE)[..., ::-1]
idet = iris_detector.IrisDetector()
idet.set_detector(fd) # fd = face_detector.FaceAlignmentDetector()
eye_landmarks = idet.detect_iris(im)
```

### 5. Face verification

InceptionResNetV1 from  [davidsandberg/facenet](https://github.com/davidsandberg/facenet) and LResNet100E-IR (ArcFace@ms1m-refine-v2) from [deepinsight/insightface](https://github.com/deepinsight/insightface) are provided as face verificaiton model. To use ArcFace model, download the weights file from [here](https://drive.google.com/uc?id=1H37LER8mRRI4q_nxpS3uQz3DcGHkTrNU) and put it under `./models/verifier/insightface/`.

```python
from models.verifier import face_verifier

im1 = cv2.imread(PATH_TO_IMAGE1)[..., ::-1]
im2 = cv2.imread(PATH_TO_IMAGE2)[..., ::-1]
fv = face_verifier.FaceVerifier(extractor="facenet") # extractor="insightface"
# fv.set_detector(fd) # fd = face_detector.FaceAlignmentDetector()
result, distance = fv.verify(im1, im2, threshold=0.5, with_detection=False, return_distance=True)
```

### 6. Gender and age estimation
```python
from models.estimator import gender_age_estimator

im = cv2.imread(PATH_TO_IMAGE)[..., ::-1]
gae = gender_age_estimator.GenderAgeEstimator()
gae.set_detector(fd) # fd = face_detector.FaceAlignmentDetector()
gender, age = gae.predict_gender_age(im, with_detection=True)
```

## Known issues
It works fine on Colab at this point (2019/06/11) but for certain Keras/TensorFlow version, it throws errors loading `2DFAN-1_keras.h5` or `2DFAN-2_keras.h5`.

## Requirements
- Keras 2.2.4
- TensorFlow 1.12.0 or 1.13.1

## Acknowledgments
We learnt a lot from [1adrianb/face-alignment](https://github.com/1adrianb/face-alignment), [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch), [swook/GazeML](https://github.com/swook/GazeML), [deepinsight/insightface](https://github.com/deepinsight/insightface), and [davidsandberg/facenet](https://github.com/davidsandberg/facenet).
