import numpy as np
import cv2
from pathlib import Path

from utils.umeyama import umeyama

FILE_PATH = Path(__file__).parent.resolve()

class GenderAgeEstimator():
    def __init__(self, model_type="insightface"):
        self.model_type = model_type
        if model_type == "insightface":
            self.input_resolution = 112
            self.weights_path = str(FILE_PATH)+"/insightface/fmobilenet_keras.h5"
        else:
            raise ValueError(f"Received an unknown model_type: {model_type}.")
            
        self.net = self.build_networks()
    
    def set_detector(self, detector):
        self.detector = detector
            
    def build_networks(self):
        if self.model_type == "insightface":
            from .insightface.fmobilenet import fmobilenet
            net = fmobilenet()
            net.load_weights(self.weights_path)
            return net
        
    def predict_gender_age(self, im, with_detection=True, return_face=False):
        if with_detection:
            try:
                if self.model_type == "insightface":
                    faces, landmarks = self.detector.detect_face(im, with_landmarks=True)
                    landmarks = [self.detector.convert_landmarks_68_to_5(l) for l in landmarks]
                else:
                    raise ValueError(f"Received an unknown model_type: {model_type}.")
            except:
                raise Exception("Error occured duaring gender/age estimation. \
                Please check if face detector has been set through set_detector().")
            if len(faces) > 1:
                print("Multiple faces detected, only the most confident one is used for gender/age estimation.")
                most_conf_idx = np.argmax(faces, axis=0)[-1]
                faces = faces[most_conf_idx:most_conf_idx+1]
                if model_type == "insightface":
                    landmarks = landmarks[most_conf_idx:most_conf_idx+1]
                    
            face = self.align_face(im, landmarks[0][..., ::-1], self.input_resolution)
        else:
            face = im            
            face = cv2.resize(face, (self.input_resolution, self.input_resolution))
        
        input_array = face[np.newaxis, ...] 
        pred = self.net.predict([input_array])
        gender, age = self.post_process(pred)
        if return_face:
            return gender, age, face
        else:
            return gender, age   
    
    @staticmethod
    def post_process(pred):
        # Refer to:
        # https://github.com/deepinsight/insightface/blob/master/gender-age/face_model.py#L89
        g = pred[:,0:2]
        gender = np.argmax(g)
        a = pred[:,2:202].reshape((100,2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))
        return gender, age
    
    @staticmethod
    def align_face(im, src, size):
        # Refer to:
        # https://github.com/deepinsight/insightface/blob/master/src/common/face_preprocess.py#L46
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041] ], dtype=np.float32 )
        dst[:,0] += 8.0        
        M = umeyama(src, dst, True)[0:2]
        warped = cv2.warpAffine(im, M, (size, size), borderValue=0.0)
        return warped 
        
        
