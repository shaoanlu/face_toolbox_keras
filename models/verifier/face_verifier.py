from keras.layers import Lambda, Input
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import numpy as np
import cv2
from scipy.spatial import distance
from pathlib import Path

#from . import umeyama
from utils.umeyama import umeyama

FILE_PATH = Path(__file__).parent.resolve()

class FaceVerifier():
    def __init__(self, extractor="facenet", classes=512):
        self.extractor_type = extractor
        self.latent_dim = classes
        if extractor == "facenet":
            self.input_resolution = 160
            self.weights_path = str(FILE_PATH)+"/facenet/facenet_keras_weights_VGGFace2.h5"
        elif extractor == "insightface":
            self.input_resolution = 112
            self.weights_path = str(FILE_PATH)+"/insightface/lresnet100e_ir_keras.h5"
        else:
            raise ValueError(f"Received an unknown extractor: {str(extractor)}.")
        
        self.net = self.build_networks(classes=classes)        
        self.net.trainable = False
        
        self.detector = False
        
    def build_networks(self, classes=512):
        if self.extractor_type == "facenet":
            """
            InceptionResNetV1 expects inputs that have range [0, 255].
            """
            from .facenet.inception_resnet_v1 import InceptionResNetV1
            input_tensor = Input((None, None, 3))
            facenet = InceptionResNetV1(weights_path=self.weights_path, classes=classes)
            facenet = Model(facenet.inputs, facenet.layers[-1].output) # layers[-1] is a BN layers
            resize_layer = self.resize_tensor(size=self.input_resolution)
            preprocess_layer = self.preprocess()
            l2_normalize = self.l2_norm()
            output_tensor = l2_normalize(facenet(preprocess_layer(resize_layer(input_tensor))))       
            return Model(input_tensor, output_tensor)
        elif self.extractor_type == "insightface":
            """
            LResNet100E-IR expects input images that have range [0, 255].
            """
            from .insightface.lresnet100e_ir import LResNet100E_IR
            input_tensor = Input((None, None, 3))
            lresnet100e_ir = LResNet100E_IR(weights_path=self.weights_path)
            resize_layer = self.resize_tensor(size=self.input_resolution)
            l2_normalize = self.l2_norm()
            output_tensor = l2_normalize(lresnet100e_ir(resize_layer(input_tensor)))        
            return Model(input_tensor, output_tensor)
    
    def set_detector(self, detector):
        self.detector = detector
        
    def resize_tensor(self, size):
        input_tensor = Input((None, None, 3)) 
        output_tensor = Lambda(lambda x: tf.image.resize_bilinear(x, [size, size]))(input_tensor)
        return Model(input_tensor, output_tensor)
        
    def preprocess(self):        
        def preprocess_facenet(x):
            """ 
                tf.image.per_image_standardization
                K.mean & K.std axis being [-3,-2,-1] or [-1,-2,-3] does nor affect output
                since the output shape is [batch_size, 1, 1, 1].
            """
            x = (x - 127.5) / 128
            x = K.map_fn(lambda im: tf.image.per_image_standardization(im), x)
            return x     
        
        input_tensor = Input((None, None, 3))      
        output_tensor = Lambda(preprocess_facenet)(input_tensor)        
        return Model(input_tensor, output_tensor)
    
    def l2_norm(self):            
        input_tensor = Input((self.latent_dim,))
        output_tensor = Lambda(lambda x: K.l2_normalize(x))(input_tensor)
        return Model(input_tensor, output_tensor)
    
    def verify(self, im1, im2, threshold=0.5, with_detection=False, return_distance=True):
        emb1 = self.extract_embeddings(im1, with_detection=with_detection)
        emb2 = self.extract_embeddings(im2, with_detection=with_detection)
        
        if self.extractor_type == "facenet":
            dist = self.compute_cosine_distance(emb1, emb2)
        elif self.extractor_type == "insightface":
            euclidean_dist = np.sum(np.square(emb1 - emb2))
            cosine_dist = 1 - np.dot(emb1, emb2.T)
            dist = (euclidean_dist + cosine_dist) / 3.75 # 3.75 here is purely heuristic so that it works fine with default threshold=0.5.
            dist = float(dist)
        is_same_person = (dist <= threshold)
        if return_distance:
            return is_same_person, dist
        else:
            return is_same_person
    
    def extract_embeddings(self, im, with_detection=False, return_face=False):
        if with_detection:
            try:
                if self.extractor_type == "facenet":
                    faces = self.detector.detect_face(im, with_landmarks=False)
                elif self.extractor_type == "insightface":
                    faces, landmarks = self.detector.detect_face(im, with_landmarks=True)
                    landmarks = [self.detector.convert_landmarks_68_to_5(l) for l in landmarks]
            except:
                raise NameError("Error occured duaring face detection. \
                Please check if face detector has been set through set_detector().")
            if len(faces) > 1:
                print("Multiple faces detected, only the most confident one is used for verification.")
                most_conf_idx = np.argmax(faces, axis=0)[-1]
                faces = faces[most_conf_idx:most_conf_idx+1]
                if self.extractor_type == "insightface":
                    landmarks = landmarks[most_conf_idx:most_conf_idx+1]
                    
            if self.extractor_type == "facenet":
                x0, y0, x1, y1, _ = faces[0].astype(np.int32)
                face = im[x0:x1, y0:y1]            
            elif self.extractor_type == "insightface":
                face = self.align_face(im, landmarks[0][..., ::-1], self.input_resolution)
        else:
            face = im
        
        input_array = face[np.newaxis, ...] 
        embeddings = self.net.predict([input_array])
        if return_face:
            return embeddings, face
        else:
            return embeddings  
    
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
    
    @staticmethod
    def compute_cosine_distance(emb1, emb2):
        return distance.cosine(emb1, emb2)
        
        