import csv
import os
import time
from itertools import islice
from threading import Thread, Lock

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget
from matplotlib import ticker

from models.concentration_evaluator import ConcentrationEvaluation
from pipeline_module.core.base_module import DictData
from ui.class_concentration import Ui_ClassConcentration

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置

from pipeline_module.classroom_action_module import ConcentrationEvaluationModule
from pipeline_module.core.task_solution import TaskSolution
from pipeline_module.pose_modules import AlphaPoseModule
from pipeline_module.video_modules import VideoModule
from pipeline_module.vis_modules import ClassConcentrationVisModule
from pipeline_module.yolo_modules import YoloV5Module
from smart_classroom.list_items import VideoSourceItem
from utils.common import second2str, OffsetList

yolov5_weight = './weights/yolov5s.torchscript.pt'
alphapose_weight = './weights/halpe136_mobile.torchscript.pth'
classroom_action_weight = './weights/classroom_action_lr_front_v2.torchscript.pth'
face_aligner_weights = 'weights/mobilenet56_se_external_model_best.torchscript.pth'
device = 'cuda'


class ClassConcentrationApp(QWidget, Ui_ClassConcentration):
    push_frame_signal = QtCore.pyqtSignal(DictData)
    draw_img_on_window_signal = QtCore.pyqtSignal(np.ndarray, QWidget)
    plot_base_h = 4.8

    def __init__(self, parent=None):
        super(ClassConcentrationApp, self).__init__(parent)
        self.setupUi(self)
        self.video_source = 0
        self.frame_data_list = OffsetList()
        self.opened_source = None
        self.playing = None
        self.playing_real_time = False
        self.pushed_frame = False

        # 视频事件
        # 设置视频源事件
        self.open_source_lock = Lock()
        self.open_source_btn.clicked.connect(
            lambda: self.open_source(self.video_source_txt.text() if len(self.video_source_txt.text()) != 0 else 0))
        self.video_resource_list.itemClicked.connect(lambda item: self.open_source(item.src))
        self.video_resource_file_list.itemClicked.connect(lambda item: self.open_source(item.src))

        self.close_source_btn.clicked.connect(self.close_source)
        self.play_video_btn.clicked.connect(self.play_video)
        self.stop_playing_btn.clicked.connect(self.stop_playing)
        self.video_process_bar.valueChanged.connect(self.change_frame)
        self.push_frame_signal.connect(self.push_frame)

        # 设置列表
        self.draw_img_on_window_signal.connect(self.draw_img_on_window2)
        # 初始化视频源
        self.init_video_source()

        # 图像坐标数据
        self.x_time_data = []
        self.y_action_data = []
        self.y_face_data = []
        self.y_head_pose_data = []
        self.y_primary_level_data = []
        self.primary_factor = None
        self.draw_img_timer = QTimer(self)
        self.draw_img_timer.timeout.connect(self.refresh_img_on_window)

        # 初始化界面剩余部分
        self.init_rest_window()

    def init_rest_window(self):
        self.real_time_catch_ico.setPixmap(QPixmap(':/videos/scan.ico'))
        pass

    def init_video_source(self):
        # 添加视频通道
        VideoSourceItem(self.video_resource_list, "摄像头", 0).add_item()
        # 添加本地视频文件
        local_source = 'resource/videos/class_concentration'
        videos = [*filter(lambda x: x.endswith('.mp4'), os.listdir(local_source))]
        for video_name in videos:
            VideoSourceItem(self.video_resource_file_list,
                            video_name,
                            os.path.join(local_source, video_name),
                            ico_src=':/videos/multimedia.ico').add_item()

        with open('resource/video_sources.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in islice(reader, 1, None):
                VideoSourceItem(self.video_resource_list, row[0], row[1],
                                ico_src=':/videos/webcam.ico').add_item()

    def open_source(self, source):
        self.open_source_lock.acquire(blocking=True)
        if self.opened_source is not None:
            self.close_source()
        # Loading
        frame = np.zeros((480, 640, 3), np.uint8)
        (f_w, f_h), _ = cv2.getTextSize("Loading", cv2.FONT_HERSHEY_TRIPLEX, 1, 2)

        cv2.putText(frame, "Loading", (int((640 - f_w) / 2), int((480 - f_h) / 2)),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1, (255, 255, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.video_screen.width() - 9, self.video_screen.height() - 9))  # 调整图像大小
        image_height, image_width, image_depth = frame.shape
        frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                       image_width * image_depth,
                       QImage.Format_RGB888)
        self.video_screen.setPixmap(QPixmap.fromImage(frame))

        # 启动视频源
        def open_source_func(self):
            fps = 12
            self.opened_source = TaskSolution() \
                .set_source_module(VideoModule(source, fps=fps)) \
                .set_next_module(YoloV5Module(yolov5_weight, device)) \
                .set_next_module(AlphaPoseModule(alphapose_weight, device)) \
                .set_next_module(ConcentrationEvaluationModule(classroom_action_weight)) \
                .set_next_module(ClassConcentrationVisModule(lambda d: self.push_frame_signal.emit(d)))
            self.opened_source.start()
            self.playing_real_time = True
            self.open_source_lock.release()

        Thread(target=open_source_func, args=[self]).start()

    def close_source(self):
        if self.opened_source is not None:
            self.stop_playing()
            self.opened_source.close()
            self.opened_source = None
            self.frame_data_list.clear()
            self.video_process_bar.setMaximum(-1)
            self.playing_real_time = False
            for l in [self.x_time_data,
                      self.y_action_data,
                      self.y_face_data,
                      self.y_head_pose_data,
                      self.y_primary_level_data]:
                l.clear()

    def push_frame(self, data):
        try:
            max_index = self.frame_data_list.max_index()
            time_process = self.frame_data_list[max_index].time_process if len(self.frame_data_list) > 0 else 0
            data.time_process = time_process + data.interval
            # 添加帧到视频帧列表
            self.frame_data_list.append(data)
            while len(self.frame_data_list) > 500:
                self.frame_data_list.pop()
            self.video_process_bar.setMinimum(self.frame_data_list.min_index())
            self.video_process_bar.setMaximum(self.frame_data_list.max_index())
            data.frame_num = max_index + 1
            # 显示图片
            if not hasattr(data, 'skipped'):
                self.add_data_to_list(data)

            # 判断是否进入实时播放状态
            if self.playing_real_time:
                self.video_process_bar.setValue(self.video_process_bar.maximum())
        except Exception as e:
            print('push_frame', e)

    def playing_video(self):
        try:
            while self.playing is not None and not self.playing_real_time:
                current_frame = self.video_process_bar.value()
                max_frame = self.video_process_bar.maximum()
                if current_frame < 0:
                    continue
                elif current_frame < max_frame:
                    data = self.frame_data_list[current_frame]
                    if current_frame < max_frame:
                        self.video_process_bar.setValue(current_frame + 1)
                    time.sleep(data.interval)
                else:
                    self.stop_playing()
                    self.playing_real_time = True
        except Exception as e:
            print(e)

    def stop_playing(self):
        if self.playing is not None:
            self.playing = None

    def add_data_to_list(self, data):
        """
        将数据绘制到界面图像得到位置
        """
        try:
            time_process = second2str(data.time_process)
            max_idx = len(self.x_time_data) - 1
            if max_idx >= 0 and time_process == self.x_time_data[max_idx]:
                return
                # 更新数据
            self.x_time_data.append(time_process)

            concentration_evaluation: ConcentrationEvaluation = data.concentration_evaluation
            secondary_mean_levels = np.mean(concentration_evaluation.secondary_levels, axis=0)

            self.y_action_data.append(secondary_mean_levels[0])
            self.y_face_data.append(secondary_mean_levels[1])
            self.y_head_pose_data.append(secondary_mean_levels[2])
            self.y_primary_level_data.append(np.mean(concentration_evaluation.primary_levels))

            while len(self.x_time_data) > self.line_data_limit_spin.value():
                for i in [self.x_time_data,
                          self.y_action_data,
                          self.y_face_data,
                          self.y_head_pose_data,
                          self.y_primary_level_data]:
                    i.pop(0)
            self.primary_factor = concentration_evaluation.primary_factor

            self.pushed_frame = True

        except Exception as e:
            print("add_data_to_list", e)

    def refresh_img_on_window(self):
        print("timer")
        if not self.pushed_frame:
            return
        print("ok")
        for y_data, img_widget, color in [
            (self.y_action_data, self.action_level_img, (1, 0, 0, 1)),
            (self.y_face_data, self.face_level_img, (0, 1, 0, 1)),
            (self.y_head_pose_data, self.head_pose_level_img, (0, 0, 1, 1)),
            (self.y_primary_level_data, self.primary_level_img, (0.4, 0.3, 0.3, 1))
        ]:
            self.draw_line_img_on_windows(y_data, img_widget, color)

        self.draw_radar_img_on_window(self.primary_factor, self.primary_factor_img)

        self.pushed_frame = False

    @staticmethod
    def draw_img_on_window2(frame, img_widget):
        frame = cv2.resize(frame,
                           (img_widget.width() - 9, img_widget.height() - 9))  # 调整图像大小
        image_height, image_width, image_depth = frame.shape
        frame = QImage(frame, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                       image_width * image_depth,
                       QImage.Format_RGBA8888)
        img_widget.setPixmap(QPixmap.fromImage(frame))

    def draw_radar_img_on_window(self, y_data, img_widget):
        try:
            # data from United Nations World Population Prospects (Revision 2019)
            # https://population.un.org/wpp/, license: CC BY 3.0 IGO
            aspect = img_widget.width() / img_widget.height()
            # 绘图
            fig = plt.figure(figsize=(self.plot_base_h * aspect, self.plot_base_h))
            ax = fig.add_subplot(111, projection='polar')
            theta = np.arange(0, 2 * np.pi, 2 * np.pi / len(y_data))
            theta = np.concatenate((theta, [theta[0]]))
            y_data = np.concatenate((y_data, [y_data[0]]))
            ax.plot(theta, y_data, 'o--')
            ax.fill(theta, y_data, alpha=0.2, color='r')
            # 设置网格标签
            ax.tick_params(labelsize=23)
            ax.set_thetagrids(theta * 180 / np.pi, ["行为", '情绪', '抬头'])  # 设置网格标签
            # matplotlib 转ndarray图像
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            self.draw_img_on_window_signal.emit(frame, img_widget)
        except Exception as e:
            print('draw_line_img_on_windows', e)

    def draw_line_img_on_windows(self, y_data, img_widget, color, xlabel='时间', ylabel='专注度'):
        try:
            # data from United Nations World Population Prospects (Revision 2019)
            # https://population.un.org/wpp/, license: CC BY 3.0 IGO
            aspect = img_widget.width() / img_widget.height()
            # 绘图
            fig = plt.figure(figsize=(self.plot_base_h * aspect, self.plot_base_h))
            ax = fig.add_subplot(111)
            ax.plot(self.x_time_data, y_data, color=color, lw=2)
            # ax.set_xlabel(xlabel, fontdict=dict(fontsize=23))
            ax.set_ylabel(ylabel, fontdict=dict(fontsize=23))
            ax.set_ylim([0, 5])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(max(2, int(self.line_data_limit_spin.value() / 5 - 1))))
            ax.tick_params(labelsize=23)
            ax.tick_params(axis='x', labelrotation=10)

            # matplotlib 转ndarray图像
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            self.draw_img_on_window_signal.emit(frame, img_widget)
        except Exception as e:
            print('draw_line_img_on_windows', e)

    def play_video(self):
        if self.playing is not None:
            return
        self.playing = Thread(target=self.playing_video, args=())
        self.playing.start()

    def change_frame(self):
        try:
            if len(self.frame_data_list) == 0:
                return
            current_frame = self.video_process_bar.value()
            max_frame = self.video_process_bar.maximum()
            self.playing_real_time = current_frame == max_frame  # 是否开启实时播放
            # 更新界面
            data = self.frame_data_list[current_frame]
            maxData = self.frame_data_list[max_frame]
            frame = data.get_draw_frame(show_box=self.show_box_ckb.isChecked())
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.video_screen.width() - 9, self.video_screen.height() - 9))  # 调整图像大小
            image_height, image_width, image_depth = frame.shape
            frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                           image_width * image_depth,
                           QImage.Format_RGB888)
            self.video_screen.setPixmap(QPixmap.fromImage(frame))
            # 显示时间
            current_time_process = second2str(data.time_process)
            max_time_process = second2str(maxData.time_process)

            self.time_process_label.setText(f"{current_time_process}/{max_time_process}")
        except Exception as e:
            print('change_frame', e)

    def close(self):
        self.draw_img_timer.stop()
        self.close_source()
        super(ClassConcentrationApp, self).close()

    def open(self):
        self.draw_img_timer.start(5000)
