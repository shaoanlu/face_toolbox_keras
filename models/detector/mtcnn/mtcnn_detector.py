import os
import numpy as np
import tensorflow as tf
from keras import backend as K

from . import mtcnn_detect_face

class MTCNN():
    """
    This class load the MTCNN network and perform face detection.
    
    Attributes:
        model_path: path to the MTCNN weights files
    """
    def __init__(self, model_path="./mtcnn/"):
        self.pnet = None
        self.rnet = None
        self.onet = None
        self._create_mtcnn(K.get_session(), model_path)
        
    def _create_mtcnn(self, sess, model_path):
        if not model_path:
            model_path, _ = os.path.split(os.path.realpath(__file__))

        with tf.variable_scope('pnet'):
            data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
            pnet = mtcnn_detect_face.PNet({'data':data})
            pnet.load(os.path.join(model_path, 'det1.npy'), sess)
        with tf.variable_scope('rnet'):
            data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
            rnet = mtcnn_detect_face.RNet({'data':data})
            rnet.load(os.path.join(model_path, 'det2.npy'), sess)
        with tf.variable_scope('onet'):
            data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
            onet = mtcnn_detect_face.ONet({'data':data})
            onet.load(os.path.join(model_path, 'det3.npy'), sess)
        self.pnet = K.function([pnet.layers['data']], [pnet.layers['conv4-2'], pnet.layers['prob1']])
        self.rnet = K.function([rnet.layers['data']], [rnet.layers['conv5-2'], rnet.layers['prob1']])
        self.onet = K.function([onet.layers['data']], [onet.layers['conv6-2'], onet.layers['conv6-3'], onet.layers['prob1']])
    
    def detect_face(self, image, minsize=20, threshold=0.7, factor=0.709):            
        faces, pnts = mtcnn_detect_face.detect_face(
            image, minsize, 
            self.pnet, self.rnet, self.onet, 
            [0.6, 0.7, threshold], 
            factor)
        #faces = self._process_mtcnn_bbox(faces, image.shape)
        bboxes = [faces[i, ...] for i in range(faces.shape[0])]
        return bboxes

    @staticmethod
    def _process_mtcnn_bbox(bboxes, im_shape):
        # output bbox coordinate of MTCNN is (y0, x0, y1, x1)
        # Here we process the bbox coord. to a square bbox with ordering (y0, x0, y1, x1)
        for i, bbox in enumerate(bboxes):
            y0, x0, y1, x1 = bboxes[i,0:4]
            w = int(y1 - y0)
            h = int(x1 - x0)
            length = (w + h)/2
            center = (int((x1+x0)/2),int((y1+y0)/2))
            new_x0 = np.max([0, (center[0]-length//2)])#.astype(np.int32)
            new_x1 = np.min([im_shape[0], (center[0]+length//2)])#.astype(np.int32)
            new_y0 = np.max([0, (center[1]-length//2)])#.astype(np.int32)
            new_y1 = np.min([im_shape[1], (center[1]+length//2)])#.astype(np.int32)
            bboxes[i,0:4] = new_y0, new_x0, new_y1, new_x1
        return bboxes
