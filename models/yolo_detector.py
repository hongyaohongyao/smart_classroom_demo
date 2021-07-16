import cv2
import numpy as np
import torch
from torchvision import transforms

from utils.general import non_max_suppression


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class YoloV5Detector:
    def __init__(self, weights, device):
        self.device = device
        self.model = torch.jit.load(weights).to(device)
        self.conf_thres = 0.35
        self.iou_thres = 0.45
        self.agnostic_nms = False
        self.max_det = 1000
        self.classes = [0]
        self.transformer = transforms.Compose([transforms.ToTensor()])
        # 预热
        _ = self.model(torch.zeros(1, 3, 640, 480).to(self.device))

    def preprocess_img(self, img):
        # img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img]
        # # Stack
        # img = np.stack(img, 0)
        # to_tensor
        # Convert
        # img = img[:, :, :, ::-1].transpose(0, 3, 1, 2) / 255  # BGR to RGB, to bsx3x416x416
        # img = np.ascontiguousarray(img)

        # if img.ndim == 3:
        #     img = np.expand_dims(img, 0)
        # img = img[:, :, :, ::-1].transpose(0, 3, 1, 2) / 255  # BGR to RGB, to bsx3x416x416
        # img = np.ascontiguousarray(img)

        return self.transformer(img[:, :, ::-1].copy()).unsqueeze(0).to(self.device, dtype=torch.float32)

    def detect(self, img):
        # img0 = img.copy()
        # 预处理
        img = self.preprocess_img(img)
        # 检测
        pred = self.model(img)[0]
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)
        # for i, det in enumerate(pred):
        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0[i].shape).round()
        pred = pred[0].detach().cpu()
        return pred
