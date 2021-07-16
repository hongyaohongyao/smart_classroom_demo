import os

import numpy as np

from silent_face.src.anti_spoof_predictor import AntiSpoofPredictor
from silent_face.src.utility import parse_model_name
from utils.img_cropper import CropImage

wanted_model_index = [0]


class SilentFaceDetector:
    def __init__(self, device_id='cpu', model_dir='weights/anti_spoof_models'):
        self.models = []
        self.params = []
        for i, model_name in enumerate(os.listdir(model_dir)):
            if i not in wanted_model_index:
                continue
            self.models.append(AntiSpoofPredictor(device_id, os.path.join(model_dir, model_name)))
            self.params.append(parse_model_name(model_name))

    def detect(self, frame, face_location):
        face_location = face_location.copy()
        face_location[2:] = face_location[2:] - face_location[:2]
        prediction = np.zeros((1, 3))
        # sum the prediction from single model's result
        for model, (h_input, w_input, model_type, scale) in zip(self.models, self.params):
            param = {
                "org_img": frame,
                "bbox": face_location,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = CropImage.crop(**param)
            prediction += model.predict(img)

        # draw result of prediction
        label = np.argmax(prediction)
        return label, prediction[0][label] / len(self.models)
