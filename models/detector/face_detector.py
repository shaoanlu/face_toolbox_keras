import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from keras import backend as K

from .s3fd.s3fd_detector import S3FD
from .landmarks_detector import FANLandmarksDetector

FILE_PATH = str(Path(__file__).parent.resolve())

class BaseFaceDetector():
    def __init__(self):
        pass
    
    def detect_face(self):
        raise NotImplementedError
    
class S3FaceDetector(BaseFaceDetector):
    def __init__(self, weights_path=FILE_PATH+"/s3fd/s3fd_keras_weights.h5"):
        self.face_detector = S3FD(weights_path)
        
    def detect_face(self, image):
        # output bbox coordinate: y0 (left), x0 (top), y1 (right), x1 (bottom)
        return self.face_detector.detect_face(image)
        
    def batch_detect_face(self, image):
        raise NotImplementedError
    
class FaceAlignmentDetector(BaseFaceDetector):
    def __init__(self, 
                 fd_weights_path=FILE_PATH+"/s3fd/s3fd_keras_weights.h5", 
                 lmd_weights_path=FILE_PATH+"/FAN/2DFAN-4_keras.h5",
                 fd_type="s3fd"):
        self.fd_type = fd_type.lower()
        if fd_type.lower() == "s3fd":
            self.fd = S3FaceDetector(fd_weights_path)
        elif fd_type.lower() == "mtcnn":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown face detector {face_detector}.")
        
        self.lmd_weights_path = lmd_weights_path
        self.lmd = None
    
    def build_FAN(self):
        self.lmd = FANLandmarksDetector(self.lmd_weights_path)
    
    def detect_face(self, image, with_landmarks=True):
        """
        Returns: 
            bbox_list: bboxes in [x0, y0, x1, y1] ordering (x is the vertical axis, the height).
            landmarks_list: landmark points having shape (68, 2) with ordering [[x0, y0], [x1, y1], ..., [x67, y67].
        """
        if self.fd_type == "s3fd":
            bbox_list = self.fd.detect_face(image)
        #elif self.fd_type == "mtcnn":
        #    bbox_list, _ = self.fd.detect_face(image)
        if len(bbox_list) == 0:
            return [], []
        #if self.fd_type == "mtcnn":
        #    bbox_list = self.preprocess_mtcnn_bbox(bbox_list)
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
    
    #@staticmethod
    #def preprocess_mtcnn_bbox(bbox_list):
    #    def bbox_coord_convert(bbox, cnovert_type="mtcnn_to_s3fd"):
    #        if cnovert_type == "mtcnn_to_s3fd":
    #            # x0y1x1y0 to y0x0y1x1
    #            x0, y1, x1, y0, score = bbox
    #            return np.array([y0, x0, y1, x1, score])
    #        
    #    for i, bbox in enumerate(bbox_list):
    #        if len(bbox.shape) == 2:
    #            bbox = bbox[0]
    #        bbox_list[i] = bbox_coord_convert(bbox, cnovert_type="mtcnn_to_s3fd")
    #    return bbox_list
        
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
        
            

    