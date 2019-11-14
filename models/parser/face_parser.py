import numpy as np
import cv2
from pathlib import Path

from .BiSeNet.bisenet import BiSeNet_keras

FILE_PATH = str(Path(__file__).parent.resolve())

class FaceParser():
    def __init__(self, path_bisenet_weights=FILE_PATH+"/BiSeNet/BiSeNet_keras.h5", detector=None):
        self.parser_net = None
        self.detector = detector
    
        self.build_parser_net(path_bisenet_weights)
        
    def build_parser_net(self, path):
        parser_net = BiSeNet_keras()
        parser_net.load_weights(path)
        self.parser_net = parser_net
        
    def set_detector(self, detector):
        self.detector = detector
    
    def remove_detector(self):
        self.detector = None
    
    def parse_face(self, im, bounding_box=None, with_detection=False):
        orig_h, orig_w = im.shape[:2]
        
        # Detect/Crop face RoI
        if bounding_box == None:
            if with_detection:
                try:
                    self.detector.fd
                except:
                    raise NameError("Error occurs during face detection: \
                    detector not found in FaceParser.")
                bboxes = self.detector.fd.detect_face(im)
                faces = []
                for bbox in bboxes:
                    left, top, right, bottom, _ = bbox
                    top, left = np.maximum(top, 0), np.maximum(left, 0)
                    bottom, right = np.minimum(bottom, orig_h), np.minimum(right, orig_w)
                    top, left, bottom, right = map(np.int32, [top, left, bottom, right])
                    faces.append(im[top:bottom, left:right, :])
            else:
                faces = [im]
        else:
            left, top, right, bottom = bounding_box
            top, left = np.maximum(top, 0), np.maximum(left, 0)
            bottom, right = np.minimum(bottom, orig_h), np.minimum(right, orig_w)
            top, left, bottom, right = map(np.int32, [top, left, bottom, right])
            faces = [im[top:bottom, left:right, :]]
        
        maps = []
        for face in faces:
            # Preprocess input face for parser networks
            orig_h, orig_w = face.shape[:2]
            inp = cv2.resize(face, (512,512))
            inp = self.normalize_input(inp)
            inp = inp[None, ...]

            # Parser networks forward pass
            # Do NOT use bilinear interp. which adds artifacts to the parsing map
            out = self.parser_net.predict([inp])[0]
            parsing_map = out.argmax(axis=-1)
            parsing_map = cv2.resize(
                parsing_map.astype(np.uint8), 
                (orig_w, orig_h), 
                interpolation=cv2.INTER_NEAREST)
            maps.append(parsing_map)
        return maps
        
    @staticmethod
    def normalize_input(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # x should be RGB with range [0, 255]
        return ((x / 255) - mean) / std
