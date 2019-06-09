from keras.layers import Lambda, Input
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import numpy as np
from scipy.spatial import distance
from pathlib import Path

FILE_PATH = str(Path(__file__).parent.resolve())

class FaceVerifier():
    def __init__(self, extractor="facenet", classes=512):
        self.extractor_type = extractor
        self.latent_dim = classes
        if extractor == "facenet":
            self.input_resolution = 160
            self.weights_path = FILE_PATH+"/facenet/facenet_keras_weights_VGGFace2.h5"
        elif extractor == "insightface":
            self.input_resolution = 112
            self.weights_path = FILE_PATH+"/insightface/lresnet100e_ir_keras.h5"
        else:
            raise ValueError(f"Received an unknown extractor: {str(extractor)}.")
        
        self.net = self.build_networks(classes=classes)        
        self.net.trainable = False
        
        self.detector = False
        
    def build_networks(self, classes=512):
        if self.extractor_type == "facenet":
            """
            FaceNet().net expects input images that have pixels normalized to [-1, +1].
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
            LResNet100E-IR expects input images that have in range [0, 255].
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
    
    def extract_embeddings(self, im, with_detection=False):
        if with_detection:
            try:
                faces = self.detector.detect_face(im, with_landmarks=False)
            except:
                raise NameError("Error occured duaring face detection. \
                Please check if face detector has been set through set_detector().")
            if len(faces) > 1:
                print("Multiple faces detected, only the most confident one is used for verification.")
                most_conf_idx = np.argmax(faces, axis=0)[-1]
                faces = faces[most_conf_idx:most_conf_idx+1]
            x0, y0, x1, y1, _ = faces[0].astype(np.int32)
            face = im[x0:x1, y0:y1]
        else:
            face = im
        
        input_array = face[np.newaxis, ...] 
        embeddings = self.net.predict([input_array])
        return embeddings    
    
    @staticmethod
    def compute_cosine_distance(emb1, emb2):
        return distance.cosine(emb1, emb2)
        
        