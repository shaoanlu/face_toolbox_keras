import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from keras import backend as K

from .s3fd.s3fd_detector import S3FD
from .mtcnn.mtcnn_detector import MTCNN
from .landmarks_detector import FANLandmarksDetector

FILE_PATH = Path(__file__).parent.resolve()

class BaseFaceDetector():
    def __init__(self):
        pass
    
    def detect_face(self):
        raise NotImplementedError

class MTCNNFaceDetector(BaseFaceDetector):
    def __init__(self, weights_path=FILE_PATH / "mtcnn"):
        self.face_detector = MTCNN(weights_path)
        
    def detect_face(self, image):
        # Output bbox coordinate has ordering (y0, x0, y1, x1)
        return self.face_detector.detect_face(image)
        
    def batch_detect_face(self, image):
        raise NotImplementedError

class S3FaceDetector(BaseFaceDetector):
    def __init__(self, weights_path=FILE_PATH / "s3fd" / "s3fd_keras_weights.h5"):
        self.face_detector = S3FD(weights_path)
        
    def detect_face(self, image):
        # Output bbox coordinate has ordering (y0, x0, y1, x1)
        return self.face_detector.detect_face(image)
        
    def batch_detect_face(self, image):
        raise NotImplementedError
    
class FaceAlignmentDetector(BaseFaceDetector):
    def __init__(self, 
                 fd_weights_path=FILE_PATH / "s3fd" / "s3fd_keras_weights.h5", 
                 lmd_weights_path=FILE_PATH / "FAN" / "2DFAN-4_keras.h5",
                 fd_type="s3fd"):
        self.fd_type = fd_type.lower()
        if fd_type.lower() == "s3fd":
            self.fd = S3FaceDetector(fd_weights_path)
        elif fd_type.lower() == "mtcnn":
            self.fd = MTCNNFaceDetector()
        else:
            raise ValueError(f"Unknown face detector {face_detector}.")
        
        self.lmd_weights_path = lmd_weights_path
        self.lmd = None
    
    def build_FAN(self):
        self.lmd = FANLandmarksDetector(self.lmd_weights_path)
    
    def detect_face(self, image, with_landmarks=True):
        """
        Returns: 
            bbox_list: bboxes in [x0, y0, x1, y1] ordering (x is the vertical axis, y the height).
            landmarks_list: landmark points having shape (68, 2) with ordering [[x0, y0], [x1, y1], ..., [x67, y67].
        """
        if self.fd_type == "s3fd":
            bbox_list = self.fd.detect_face(image)
        elif self.fd_type == "mtcnn":
            bbox_list = self.fd.detect_face(image)
        if len(bbox_list) == 0:
            return [], []
            
        if with_landmarks:
            if self.lmd == None:
                print("Building FAN for landmarks detection...")
                self.build_FAN()
                print("Done.")
            landmarks_list = []
            for bbox in bbox_list:
                pnts = self.lmd.detect_landmarks(image, bounding_box=bbox)[-1]
                landmarks_list.append(np.array(pnts))
            landmarks_list = [self.post_process_landmarks(landmarks) for landmarks in landmarks_list]
            bbox_list = self.preprocess_s3fd_bbox(bbox_list)
            return bbox_list, landmarks_list
        else:
            bbox_list = self.preprocess_s3fd_bbox(bbox_list)
            return bbox_list
    
    def batch_detect_face(self, images, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def preprocess_s3fd_bbox(bbox_list):
        # Convert coord (y0, x0, y1, x1) to (x0, y0, x1, y1)
        return [np.array([bbox[1], bbox[0], bbox[3], bbox[2], bbox[4]]) for bbox in bbox_list]
        
    @staticmethod
    def post_process_landmarks(landmarks):
        # Process landmarks to have shape [68, 2]
        lms = landmarks.reshape(68, 2)[:,::-1]
        return lms
    
    @staticmethod
    def draw_landmarks(image, landmarks, color=(0, 255, 0), stroke=3):        
        for i in range(len(landmarks)): 
            x, y = landmarks[i]
            image = cv2.circle(image.copy(), (int(y), int(x)), stroke, color, -1)        
        return image
    
    @staticmethod
    def convert_landmarks_68_to_5(landmarks):
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        nose_tip = landmarks[30]
        left_mouth = landmarks[48]
        right_mouth = landmarks[54]
        new_landmarks = np.stack([
            left_eye, 
            right_eye, 
            nose_tip, 
            left_mouth, 
            right_mouth])
        return new_landmarks

