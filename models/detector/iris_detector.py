import numpy as np
import cv2
from pathlib import Path

from .ELG.elg_keras import KerasELG

FILE_PATH = str(Path(__file__).parent.resolve())
NET_INPUT_SHAPE = (108, 180)

class IrisDetector():
    def __init__(self, path_elg_weights=FILE_PATH+"/ELG/elg_keras.h5"):
        self.elg = None
        self.detector = None
        self.path_elg_weights = path_elg_weights
        
        self.build_ELG()
        
    def build_ELG(self):
        self.elg = KerasELG()
        self.elg.net.load_weights(self.path_elg_weights)
        
    def set_detector(self, detector):
        self.detector = detector
        
    def detect_iris(self, im, landmarks=None):
        """
        Input:
            im: RGB image
        Outputs:
            output_eye_landmarks: list of eye landmarks having shape (2, 18, 2) with ordering (L/R, landmarks, x/y).
        """
            
        if landmarks == None:
            try:    
                faces, landmarks = self.detector.detect_face(im, with_landmarks=True)     
            except:
                raise NameError("Error occured during face detection. Maybe face detector has not been set.")
                
        left_eye_idx = slice(36, 42)
        right_eye_idx = slice(42, 48)
        output_eye_landmarks = []
        for lm in landmarks:
            left_eye_im, left_x0y0 = self.get_eye_roi(im, lm[left_eye_idx])
            right_eye_im, right_x0y0 = self.get_eye_roi(im, lm[right_eye_idx])
            inp_left = self.preprocess_eye_im(left_eye_im)
            inp_right = self.preprocess_eye_im(right_eye_im)
            
            input_array = np.concatenate([inp_left, inp_right], axis=0)            
            pred_left, pred_right = self.elg.net.predict(input_array)
            
            lms_left = self.elg._calculate_landmarks(pred_left, eye_roi=left_eye_im)
            lms_right = self.elg._calculate_landmarks(pred_right, eye_roi=right_eye_im)
            eye_landmarks = np.concatenate([lms_left, lms_right], axis=0)
            eye_landmarks = eye_landmarks + np.array([left_x0y0, right_x0y0]).reshape(2,1,2)
            output_eye_landmarks.append(eye_landmarks)
        return output_eye_landmarks
    
    @staticmethod
    def get_eye_roi(im, lms, ratio_w=1.5):
        def adjust_hw(hw, ratio_w=1.5):
            """
            set RoI height and width to the same ratio of NET_INPUT_SHAPE
            """
            h, w = hw[0], hw[1]
            new_w = w * ratio_w
            new_h = NET_INPUT_SHAPE[0] / NET_INPUT_SHAPE[1] * new_w
            return np.array([new_h, new_w])
        h, w = im.shape[:2]
        min_xy = np.min(lms, axis=0)
        max_xy = np.max(lms, axis=0)
        hw = max_xy - min_xy
        hw = adjust_hw(hw, ratio_w=ratio_w)
        center = np.mean(lms, axis=0)
        x0, y0 = center - (hw) / 2
        x1, y1 = center + (hw) / 2
        x0, y0, x1, y1 = map(np.int32,[x0, y0, x1, y1])
        x0, y0 = np.maximum(x0, 0), np.maximum(y0, 0)
        x1, y1 = np.minimum(x1, h), np.minimum(y1, w)
        eye_im = im[x0:x1, y0:y1]
        return eye_im, (x0, y0)
    
    @staticmethod
    def preprocess_eye_im(im):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = cv2.equalizeHist(im)
        im = cv2.resize(im, (NET_INPUT_SHAPE[1], NET_INPUT_SHAPE[0]))[np.newaxis, ..., np.newaxis]
        im = im / 255 * 2 - 1
        return im
    
    @staticmethod
    def draw_pupil(im, lms, stroke=3):
        draw = im.copy()
        #draw = cv2.resize(draw, (inp_im.shape[2], inp_im.shape[1]))
        pupil_center = np.zeros((2,))
        pnts_outerline = []
        pnts_innerline = []
        for i, lm in enumerate(np.squeeze(lms)):
            x, y = int(lm[0]), int(lm[1])

            if i < 8:
                draw = cv2.circle(draw, (y, x), stroke, (125,255,125), -1)
                pnts_outerline.append([y, x])
            elif i < 16:
                draw = cv2.circle(draw, (y, x), stroke, (125,125,255), -1)
                pnts_innerline.append([y, x])
                pupil_center += (y,x)
            elif i < 17:
                pass
                #draw = cv2.drawMarker(draw, (y, x), (255,200,200), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=stroke, line_type=cv2.LINE_AA)
            else:
                pass
                #draw = cv2.drawMarker(draw, (y, x), (255,125,125), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=stroke, line_type=cv2.LINE_AA)
        pupil_center = (pupil_center/8).astype(np.int32)
        draw = cv2.cv2.circle(draw, (pupil_center[0], pupil_center[1]), stroke, (255,255,0), -1)        
        draw = cv2.polylines(draw, [np.array(pnts_outerline).reshape(-1,1,2)], isClosed=True, color=(125,255,125), thickness=stroke//2)
        draw = cv2.polylines(draw, [np.array(pnts_innerline).reshape(-1,1,2)], isClosed=True, color=(125,125,255), thickness=stroke//2)
        return draw
        