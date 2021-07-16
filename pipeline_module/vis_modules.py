import copy
import time
from abc import abstractmethod
from queue import Empty

import cv2
import numpy as np
import torch
from PIL import ImageFont, Image, ImageDraw
from PyQt5.QtGui import QPixmap, QImage

from pipeline_module.core.base_module import BaseModule, TASK_DATA_OK, DictData
from utils.vis import draw_keypoints136

box_color = (0, 255, 0)
cheating_box_color = (0, 0, 255)


def draw_frame(data, draw_keypoints=False, fps=-1):
    frame = data.frame.copy()
    pred = data.detections
    preds_kps = data.keypoints
    preds_scores = data.keypoints_scores
    if pred.shape[0] > 0:
        # 绘制骨骼关键点
        if draw_keypoints and preds_kps is not None:
            draw_keypoints136(frame, preds_kps, preds_scores)
        # 绘制目标检测框和动作分类
        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil)
        for det, class_prob, best_pred in zip(pred, data.classes_probs, data.best_preds):
            det = det.to(torch.int)
            class_name = data.classes_names[best_pred]
            # show_text = f"{class_name}: %.2f" % class_prob[best_pred]
            show_text = f"{class_name}"
            show_color = box_color if best_pred == 0 else cheating_box_color
            draw.rectangle((det[0], det[1], det[2], det[3]), outline=show_color, width=2)
            # 文字
            fontText = ImageFont.truetype("resource/font/NotoSansCJKkr-Black.otf",
                                          int(40 * (min(det[2] - det[0], det[3] - det[1])) / 200),
                                          encoding="utf-8")
            draw.text((det[0], det[1]), show_text, show_color, font=fontText)
            # cv2.putText(frame, show_text,
            #             (det[0], det[1]),
            #             cv2.FONT_HERSHEY_COMPLEX,
            #             float((det[2] - det[0]) / 200),
            #             show_color)
        frame = np.asarray(frame_pil)
        # 头部姿态估计轴
        for (r, t) in data.head_pose:
            data.draw_axis(frame, r, t)
    # 绘制fps
    cv2.putText(frame, "FPS: %.2f" % fps, (0, 52), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    data.frame_anno = frame  # 保存绘制过的图像


class DrawModule(BaseModule):
    def __init__(self):
        super(DrawModule, self).__init__()
        self.last_time = time.time()

    def process_data(self, data):
        current_time = time.time()
        fps = 1 / (current_time - self.last_time)
        frame = data.frame
        pred = data.detections
        preds_kps = data.keypoints
        preds_scores = data.keypoints_scores
        for det in pred:
            show_text = "person: %.2f" % det[4]
            cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), box_color, 2)
            cv2.putText(frame, show_text,
                        (det[0], det[1]),
                        cv2.FONT_HERSHEY_COMPLEX,
                        float((det[2] - det[0]) / 200),
                        box_color)
        if preds_kps is not None:
            draw_keypoints136(frame, preds_kps, preds_scores)
        # 记录fps
        cv2.putText(frame, "FPS: %.2f" % fps, (0, 52), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        # 显示图像
        cv2.imshow("yolov5", frame)
        cv2.waitKey(40)
        self.last_time = current_time
        return TASK_DATA_OK

    def open(self):
        super(DrawModule, self).open()
        pass


class FrameDataSaveModule(BaseModule):
    def __init__(self, app):
        super(FrameDataSaveModule, self).__init__()
        self.last_time = time.time()
        self.app = app

    def process_data(self, data):
        current_time = time.time()
        fps = 1 / (current_time - self.last_time)
        frame = data.frame
        pred = data.detections
        preds_kps = data.keypoints
        preds_scores = data.keypoints_scores
        for det in pred:
            show_text = "person: %.2f" % det[4]
            cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), box_color, 2)
            cv2.putText(frame, show_text,
                        (det[0], det[1]),
                        cv2.FONT_HERSHEY_COMPLEX,
                        float((det[2] - det[0]) / 200),
                        box_color)
        if preds_kps is not None:
            draw_keypoints136(frame, preds_kps, preds_scores)
        # 记录fps
        cv2.putText(frame, "FPS: %.2f" % fps, (0, 52), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        self.app.video_screen.setPixmap(self.cvImg2qtPixmap(frame))
        time.sleep(0.04)
        self.last_time = current_time
        return TASK_DATA_OK

    @staticmethod
    def cvImg2qtPixmap(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, image_depth = frame.shape
        frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                       image_width * image_depth,
                       QImage.Format_RGB888)
        return QPixmap.fromImage(frame)

    def open(self):
        super(FrameDataSaveModule, self).open()
        pass


class DataDealerModule(BaseModule):
    def __init__(self, push_frame_func, interval=0.06, skippable=False):
        super(DataDealerModule, self).__init__(skippable=skippable)
        self.last_time = time.time()
        self.push_frame_func = push_frame_func
        self.last_data = None
        self.interval = interval
        self.size_waiting = True

        #
        self.queue_threshold = 10

    @abstractmethod
    def deal_skipped_data(self, data: DictData, last_data: DictData) -> DictData:
        pass

    @abstractmethod
    def draw_frame(self, data, fps):
        pass

    def process_data(self, data):
        if hasattr(data, 'skipped') and self.last_data is not None:
            data = self.deal_skipped_data(data, copy.copy(self.last_data))
        else:
            self.last_data = data
        current_time = time.time()
        interval = (current_time - self.last_time)
        fps = 1 / interval
        self.draw_frame(data, fps=fps)
        data.interval = interval
        data.fps = fps
        self.last_time = current_time  # 更新时间
        self.push_frame_func(data)
        if hasattr(data, 'source_fps'):
            time.sleep(1 / data.source_fps * (1 + self.self_balance_factor()))
        else:
            time.sleep(self.interval)
        return TASK_DATA_OK

    def self_balance_factor(self):
        factor = max(-0.999, (self.queue.qsize() / 20 - 0.5) / -0.5)
        # print(factor)
        return factor

    def product_task_data(self):
        # print(self.queue.qsize(), self.size_waiting)
        if self.queue.qsize() == 0:
            self.size_waiting = True
        if self.queue.qsize() > self.queue_threshold or not self.size_waiting:
            self.size_waiting = False
            try:
                task_data = self.queue.get(block=True, timeout=1)
                return task_data
            except Empty:
                return self.ignore_task_data
        else:
            time.sleep(1)
            return self.ignore_task_data

    def put_task_data(self, task_data):
        self.queue.put(task_data)

    def open(self):
        super(DataDealerModule, self).open()
        pass


class CheatingDetectionVisModule(DataDealerModule):

    def __init__(self, push_frame_func, interval=0.06, skippable=False):
        super(CheatingDetectionVisModule, self).__init__(push_frame_func, interval, skippable)

    def deal_skipped_data(self, data: DictData, last_data: DictData) -> DictData:
        frame = data.frame
        data = last_data
        data.frame = frame
        data.detections = data.detections.clone()
        # 添加抖动
        data.detections[:, :4] += torch.rand_like(data.detections[:, :4]) * 3
        return data

    def draw_frame(self, data, fps):
        draw_frame(data, fps=fps)


class DynamicAttendanceVisModule(DataDealerModule):

    def __init__(self, push_frame_func, known_names, interval=0.06, skippable=False):
        super(DynamicAttendanceVisModule, self).__init__(push_frame_func, interval, skippable)
        self.known_names = known_names

    def deal_skipped_data(self, data: DictData, last_data: DictData) -> DictData:
        frame = data.frame
        data = last_data
        data.frame = frame
        data.face_locations = np.copy(data.face_locations)
        # 添加抖动
        data.face_locations[:, :4] += np.random.rand(*data.face_locations[:, :4].shape) * 3
        return data

    def draw_frame(self, data, fps):
        def opt_draw_frame(show_raw, show_locations, threshold, data=data, self=self):
            frame = data.frame.copy()
            face_locations = data.face_locations
            raw_face_labels = data.raw_face_labels
            raw_face_probs = data.raw_face_probs
            face_labels = data.face_labels
            face_probs = data.face_probs
            #
            frame_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(frame_pil)  # 创建画板
            if show_locations:
                for (x1, y1, x2, y2), \
                    face_label, face_prob, \
                    raw_face_label, raw_face_prob in zip(face_locations,
                                                         face_labels, face_probs,
                                                         raw_face_labels,
                                                         raw_face_probs):
                    color = (0, 0, 255) if face_prob > threshold else (255, 0, 0)
                    # 把人脸框出来标号
                    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

                    fontText = ImageFont.truetype("resource/font/NotoSansCJKkr-Black.otf",
                                                  int(40 * (min(x2 - x1, y2 - y1)) / 200),
                                                  encoding="utf-8")
                    # 显示处理冲突后的检测结果
                    if face_prob < threshold:
                        label_name = f'{self.known_names[face_label]}:{(1 - face_prob) * 100:8.2f}%'
                    else:
                        label_name = '请正视摄像头'
                    f_w, f_h = fontText.getsize(label_name)
                    draw.rectangle([(x1, y1 - f_h), (x2, y1)], color)
                    draw.text((x1, y1 - f_h), label_name, (255, 255, 255), font=fontText)

                    # 显示原始检测结果
                    if show_raw:
                        raw_label_name = f'{self.known_names[raw_face_label]}:{(1 - raw_face_prob) * 100:8.2f}%'
                        f_w_2, f_h_2 = fontText.getsize(raw_label_name)
                        draw.rectangle([(x1, y1 - f_h_2 - f_h), (x2, y1 - f_h)], (0, 200, 200))
                        draw.text((x1, y1 - f_h_2 - f_h), raw_label_name, (255, 255, 255), font=fontText)
            return np.array(frame_pil)

        data.get_draw_frame = lambda show_raw=True, show_locations=True, threshold=0.25: opt_draw_frame(show_raw,
                                                                                                        show_locations,
                                                                                                        threshold)
