import csv
import os
import time
from itertools import islice
from threading import Lock, Thread

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QWidget

from pipeline_module.core.base_module import DictData
from pipeline_module.core.task_solution import TaskSolution
from pipeline_module.face_detection_module import FaceDetectionModule
from pipeline_module.face_encoding_module import FaceEncodingModule
from pipeline_module.face_match_module import FaceMatchModule
from pipeline_module.video_modules import VideoModule
from pipeline_module.vis_modules import DynamicAttendanceVisModule
from smart_classroom.list_items import VideoSourceItem, FaceListItem, AttendanceItemWrapper
from ui.dynamic_attendance import Ui_DynamicAttendance
from utils.common import OffsetList, second2str, read_img_from_cn_path, read_encoding_json2npy

face_bank_base_dir = 'resource/face_bank'


class DynamicAttendanceApp(QWidget, Ui_DynamicAttendance):
    init_attendance_task_signal = QtCore.pyqtSignal()
    push_frame_signal = QtCore.pyqtSignal(DictData)
    update_attendance_task_signal = QtCore.pyqtSignal(DictData)

    def __init__(self, parent=None):
        super(DynamicAttendanceApp, self).__init__(parent)
        self.setupUi(self)
        self.video_source = 0
        self.frame_data_list = OffsetList()
        self.opened_source = None
        self.playing = None
        self.playing_real_time = False

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
        self.early_stop_video_btn.clicked.connect(self.close_ahead)
        # 设置其他信号槽事件
        self.show_raw_lbl_ckb.stateChanged.connect(self.change_frame)
        self.show_anno_ckb.stateChanged.connect(self.change_frame)
        self.face_match_threshold_dspin.valueChanged.connect(
            lambda: (self.change_frame(), self.update_attendance_list_widget())
        )
        self.init_attendance_task_signal.connect(self.init_attendance_task)
        self.update_attendance_task_signal.connect(self.update_attendance_task_list)
        # 初始化视频源
        self.init_video_source()
        # 初始化人脸数据库
        self.init_face_bank()

        # 其他事件
        # 签到表定位

        def local_to_cheater(x):
            self.stop_playing()
            if x.frame_num > 0:
                self.video_process_bar.setValue(x.frame_num)

        self.attended_list.itemClicked.connect(local_to_cheater)

        # 过滤
        def student_list_filter(txt: str, self=self):
            txt = txt.strip()
            if txt == '':
                for i in range(self.student_list.count()):
                    self.student_list.item(i).setHidden(False)
            else:
                for i in range(self.student_list.count()):
                    item = self.student_list.item(i)
                    item.setHidden(item.name.find(txt) < 0)

        self.student_list_filter_txt.textChanged.connect(student_list_filter)

        self.class_list_filter_txt.textChanged.connect(lambda: self.refresh_face_bank())

    def refresh_face_bank(self):
        """
        刷新人脸库
        """
        try:
            self.face_bank_list_cbx.clear()
            txt = self.class_list_filter_txt.text()
            for bank_name in self.face_banks:
                if txt == '' or bank_name.find(txt) > -1:
                    self.face_bank_list_cbx.addItem(bank_name)
        except Exception as e:
            print('refresh_face_bank:', e)

    def init_face_bank(self):
        """
        初始化人脸库
        """
        try:
            self.face_banks = os.listdir(face_bank_base_dir)
            self.refresh_face_bank()
            self.face_bank_list_cbx.currentTextChanged.connect(self.open_face_bank)
            # 初始化
            self.refresh_face_bank_btn.clicked.connect(lambda: self.open_face_bank())
            self.refresh_face_bank_btn.setIcon(QIcon(':/func/refresh.ico'))
        except Exception as e:
            print('init_face_bank:', e)

    def init_video_source(self):
        """
        初始化视频源
        """
        # 添加视频通道
        VideoSourceItem(self.video_resource_list, "摄像头", 0).add_item()
        # 添加本地视频文件
        local_source = 'resource/videos/dynamic_attendance'
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
            # 初始化签到任务
            self.init_attendance_task_signal.emit()
            fps = 12
            self.opened_source = TaskSolution() \
                .set_source_module(VideoModule(source, fps=fps)) \
                .set_next_module(FaceDetectionModule()) \
                .set_next_module(FaceEncodingModule()) \
                .set_next_module(FaceMatchModule(np.array([encoding for (_, encoding, _) in self.known_faces_data]))) \
                .set_next_module(DynamicAttendanceVisModule(lambda d: self.push_frame_signal.emit(d),
                                                            self.known_face_names))
            self.opened_source.start()
            self.playing_real_time = True
            self.open_source_lock.release()

        Thread(target=open_source_func, args=[self]).start()

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

            # 动态点名
            data.frame_num = max_index + 1
            self.update_attendance_task_signal.emit(data)

            # 判断是否进入实时播放状态
            if self.playing_real_time:
                self.video_process_bar.setValue(self.video_process_bar.maximum())
        except Exception as e:
            print("push_frame", e)

    def init_attendance_task(self):
        """
        初始化签到任务
        """
        self.absented_list.clear()
        self.attended_list.clear()
        self.attendance_task_list = [AttendanceItemWrapper(self.attended_list,
                                                           self.absented_list,
                                                           face_img,
                                                           name) for (name, _, face_img) in self.known_faces_data]
        self.update_attendance_list_widget()

    def update_attendance_task_list(self, data):
        """
        更新签到任务数据
        :param data: 处理后的数据
        """
        if hasattr(data, 'skipped'):
            return
        try:
            face_labels = data.face_labels
            face_probs = data.face_probs
            face_locations = data.face_locations
            frame = data.frame
            change_flag = False
            for face_label, \
                face_prob, \
                face_location in zip(face_labels,
                                     face_probs,
                                     face_locations):
                item = self.attendance_task_list[face_label]
                if item.set_matched_data(frame, 100 * (1 - face_prob), data.frame_num, face_location):
                    item.show_on_attend_list(self.face_match_threshold_dspin.value())
                    change_flag = True
            if change_flag:
                self.update_attended_students_num_lbl()  # 更新签到学生数量显示
            if self.absented_list.count() <= 0 and self.auto_close_ahead_ckb.isChecked():
                self.close_ahead()
        except Exception as e:
            print("update_attendance_task_list", e)

    def update_attended_students_num_lbl(self):
        """
        更新签到学生数量显示
        """
        attended_student_num = self.attended_list.count()
        all_student_num = self.absented_list.count() + attended_student_num
        self.student_num_lbl.setText(f'人数:{attended_student_num}/{all_student_num}')

    def update_attendance_list_widget(self):
        """
        更新签到任务相关的列表组件
        """
        try:
            for item in self.attendance_task_list:
                item: AttendanceItemWrapper
                item.show_on_attend_list(self.face_match_threshold_dspin.value())
                pass
            self.update_attended_students_num_lbl()
        except Exception as e:
            print("update_attendance_list_widget", e)

    def close_source(self):
        try:
            if self.opened_source is not None:
                self.stop_playing()
                self.opened_source.close()
                self.opened_source = None
                self.frame_data_list.clear()
                self.video_process_bar.setMaximum(-1)
                self.playing_real_time = False
                self.attended_list.clear()
                self.absented_list.clear()
        except Exception as e:
            print('close_source', e)

    def close_ahead(self):
        if self.opened_source is not None:
            self.opened_source.close()

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
            print('playing_video', e)

    def stop_playing(self):
        if self.playing is not None:
            self.playing = None

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
            frame = data.get_draw_frame(show_raw=self.show_raw_lbl_ckb.isChecked(),
                                        show_locations=self.show_anno_ckb.isChecked(),
                                        threshold=1 - self.face_match_threshold_dspin.value() / 100)
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
        self.close_source()

    def open(self):
        self.open_face_bank()

    def open_face_bank(self, bank_name=None):
        try:
            self.student_list.clear()
            bank_name = bank_name if bank_name is not None else self.face_bank_list_cbx.currentText()
            face_bank_dir = os.path.join(face_bank_base_dir, bank_name)
            self.known_face_names, self.known_faces_data = self.get_known_faces_data(face_bank_dir)
            for idx, (name, encoding, face_img) in enumerate(self.known_faces_data):
                FaceListItem(self.student_list,
                             face_img,
                             encoding,
                             name,
                             idx,
                             face_bank_dir,
                             None
                             ).add_item()
        except Exception as e:
            print('open_face_bank: ', e)

    @staticmethod
    def get_known_faces_data(facebank):
        try:
            known_face_names = os.listdir(facebank)  # 读取已经录入的人名
            known_faces_data = [(name, read_encoding_json2npy(
                os.path.join(facebank, name, 'encoding.json')
            ), read_img_from_cn_path(
                os.path.join(facebank, name, 'face.jpg')
            )) for name in known_face_names]  # 人名，编码，图片

            return known_face_names, known_faces_data
        except Exception as e:
            print('get_known_faces_data', e)
