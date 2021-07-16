import os
import shutil

import cv2
import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QListWidget, QInputDialog, QLineEdit, QMessageBox

from ui.attendance_item import Ui_AttendanceItem
from ui.cheating_list_item import Ui_CheatingListItem
from ui.face_list_item import Ui_FaceListItem
from ui.real_time_catch import Ui_RealTimeCatch
from utils.common import second2str, validFileName
from utils.img_cropper import CropImage


class FrameData(QListWidgetItem):

    def __init__(self, list_widget: QListWidget, data, filter_idx=0):
        super(FrameData, self).__init__()
        self.list_widget = list_widget
        self.data_ = data
        self.frame_num = data.frame_num
        self.widget = FrameData.Widget(list_widget)
        self.time_process = second2str(data.time_process)

        color1 = '#ff0000' if data.num_of_passing > 0 else '#ffffff'
        color2 = '#ff0000' if data.num_of_peep > 0 else '#ffffff'
        color3 = '#ff0000' if data.num_of_gazing_around > 0 else '#ffffff'
        self.str = f"时间[{self.time_process}] " \
                   f"<span style=\" color: {color1};\">传纸条: {data.num_of_passing}</span> " \
                   f"<span style=\" color: {color2};\">低头偷看: {data.num_of_peep}</span> " \
                   f"<span style=\" color: {color3};\">东张西望: {data.num_of_gazing_around}</span>"
        self.widget.lbl.setText(self.str)
        idx = filter_idx
        if idx == 0:
            self.setHidden(True)
        elif idx == 1:
            self.setHidden(self.data_.num_of_passing == 0)
        elif idx == 2:
            self.setHidden(self.data_.num_of_peep == 0)
        elif idx == 3:
            self.setHidden(self.data_.num_of_gazing_around == 0)

        self.setSizeHint(QSize(400, 46))

    def add_item(self):
        size = self.sizeHint()
        self.list_widget.addItem(self)  # 添加
        self.widget.setSizeIncrement(size.width(), size.height())
        self.list_widget.setItemWidget(self, self.widget)

    class Widget(QWidget, Ui_CheatingListItem):
        def __init__(self, parent=None):
            super(FrameData.Widget, self).__init__(parent)
            self.setupUi(self)

    # class Widget(QLabel):
    #     def __init__(self, parent=None):
    #         super(FrameData.Widget, self).__init__(parent)


class RealTimeCatchItem(QListWidgetItem):
    def __init__(self, list_widget: QListWidget, img, detection, time_process, cheating_type, frame_num):
        super(RealTimeCatchItem, self).__init__()
        self.list_widget = list_widget
        self.widget = RealTimeCatchItem.Widget(list_widget)
        self.setSizeHint(QSize(200, 200))

        self.img = img
        self.time_process = time_process
        self.cheating_type = cheating_type
        self.frame_num = frame_num
        self.detection = detection

    def add_item(self):
        size = self.sizeHint()
        self.list_widget.insertItem(0, self)  # 添加
        self.widget.setSizeIncrement(size.width(), size.height())
        self.list_widget.setItemWidget(self, self.widget)
        # 设置图像
        catch_img = self.widget.catch_img
        frame = CropImage.crop(self.img, self.detection, 1, catch_img.width(), catch_img.height())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, image_depth = frame.shape
        frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                       image_width * image_depth,
                       QImage.Format_RGB888)
        self.widget.catch_img.setPixmap(QPixmap.fromImage(frame))
        # 设置时间
        self.widget.time_lbl.setText(f'{second2str(self.time_process)}')
        self.widget.cheating_type_lbl.setText(self.cheating_type)

    class Widget(QWidget, Ui_RealTimeCatch):
        def __init__(self, parent=None):
            super(RealTimeCatchItem.Widget, self).__init__(parent)
            self.setupUi(self)


class VideoSourceItem(QListWidgetItem):
    def __init__(self, list_widget, name, src, ico_src=":/videos/web-camera.ico"):
        super(VideoSourceItem, self).__init__()
        icon = QIcon()
        icon.addPixmap(QPixmap(ico_src), QIcon.Normal, QIcon.Off)
        self.setIcon(icon)
        self.setText(name)
        self.src = src
        self.list_widget = list_widget

    def add_item(self):
        self.list_widget.addItem(self)


class FaceListItem(QListWidgetItem):
    def __init__(self, list_widget: QListWidget,
                 img, encoding, name, idx, face_bank_dir, know_names: list = None):
        super(FaceListItem, self).__init__()
        self.list_widget = list_widget
        self.widget = FaceListItem.Widget(list_widget)
        self.setSizeHint(QSize(117, 139))

        self.img = img
        self.encoding = encoding
        self.name = name
        self.idx = idx
        self.face_bank_dir = face_bank_dir
        self.know_names = know_names

        face_img = self.widget.face_img
        frame = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (face_img.width() - 5, face_img.height() - 5))  # 调整图像大小
        image_height, image_width, image_depth = frame.shape
        frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                       image_width * image_depth,
                       QImage.Format_RGB888)
        face_img.setPixmap(QPixmap.fromImage(frame))

        self.widget.name_lbl.setText(name)
        self.widget.edit_name_btn.setIcon(QIcon(':/func/edit.ico'))
        self.widget.edit_name_btn.clicked.connect(lambda: self.edit_name())
        self.widget.edit_name_btn.setHidden(self.know_names is None)

    def edit_name(self):
        try:
            new_name, okPressed = QInputDialog.getText(self.widget, "名称修改", "名称:", QLineEdit.Normal, "")
            new_name = new_name.strip()
            if okPressed and self.check_input_name(new_name):
                # 重命名文件
                src_dir = os.path.join(self.face_bank_dir, self.name)
                dst_dir = os.path.join(self.face_bank_dir, new_name)
                os.rename(src_dir, dst_dir)
                # 修改组件信息
                self.know_names.remove(self.name)
                self.name = new_name
                self.widget.name_lbl.setText(self.name)
            elif okPressed:
                QMessageBox.information(self.widget,
                                        "未注册成功",
                                        "不要输入已经存在的名字或者不输入，名称只能包含中文、字母和数字",
                                        QMessageBox.Yes | QMessageBox.No)
        except Exception as e:
            print('edit_name:', e)

    def delete_name(self):
        try:
            file_dir = os.path.join(self.face_bank_dir, self.name)
            if os.path.exists(file_dir):
                shutil.rmtree(file_dir)
        except Exception as e:
            print('delete_name:', e)

    def check_input_name(self, name):
        return name not in self.know_names and validFileName(name)

    def add_item(self):
        size = self.sizeHint()
        self.list_widget.addItem(self)  # 添加
        self.widget.setSizeIncrement(size.width(), size.height())
        self.list_widget.setItemWidget(self, self.widget)

    class Widget(QWidget, Ui_FaceListItem):
        def __init__(self, parent=None):
            super(FaceListItem.Widget, self).__init__(parent)
            self.setupUi(self)


class AttendanceItem(QListWidgetItem):
    def __init__(self, list_widget: QListWidget,
                 face_img, match_img, face_location, prob, name):
        super(AttendanceItem, self).__init__()
        self.list_widget = list_widget
        self.name = name
        self.face_img = face_img
        self.face_name = name
        self.set_matched_data(match_img, face_location, prob, -1)
        self.setSizeHint(QSize(300, 90))

    def set_matched_data(self, matched_face_img, face_location, prob, frame_num):
        self.frame_num = frame_num
        self.matched_face_img = matched_face_img
        if face_location is not None:
            self.face_location = np.copy(face_location)
            self.face_location[2:] = self.face_location[2:] - self.face_location[:2]
        else:
            self.face_location = None
        self.prob = prob

    def add_item(self, list_widget: QListWidget = None):
        if self.list_widget is not None:
            self.removeItem()
        if list_widget is not None:
            self.list_widget = list_widget
        size = self.sizeHint()
        self.list_widget.addItem(self)  # 添加
        self.widget = AttendanceItem.Widget()
        self.widget.setSizeIncrement(size.width(), size.height())
        self.list_widget.setItemWidget(self, self.widget)

        # 设置文字和图像
        self.set_face_img(self.face_img)
        self.widget.name_lbl.setText(self.face_name)
        # 设置图像
        if self.matched_face_img is not None:
            self.set_matched_face_img()
        else:
            self.widget.matched_face_img.setHidden(True)
            self.widget.match_lbl.setHidden(True)
            self.widget.similar_degree_lbl.setHidden(True)

    def removeItem(self):
        return self.list_widget.takeItem(self.list_widget.row(self))

    def set_face_img(self, face_frame):
        face_img = self.widget.face_img
        frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (face_img.width() - 5, face_img.height() - 5))  # 调整图像大小
        image_height, image_width, image_depth = frame.shape
        frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                       image_width * image_depth,
                       QImage.Format_RGB888)
        face_img.setPixmap(QPixmap.fromImage(frame))

    def set_matched_face_img(self):
        face_img = self.widget.matched_face_img
        frame = cv2.cvtColor(CropImage.crop(self.matched_face_img,
                                            self.face_location, 1,
                                            face_img.width(),
                                            face_img.height()), cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (face_img.width() - 5, face_img.height() - 5))  # 调整图像大小
        image_height, image_width, image_depth = frame.shape
        frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                       image_width * image_depth,
                       QImage.Format_RGB888)
        face_img.setPixmap(QPixmap.fromImage(frame))
        self.widget.similar_degree_lbl.setText(f'{self.prob:8.2f}')

    class Widget(QWidget, Ui_AttendanceItem):
        def __init__(self, parent=None):
            super(AttendanceItem.Widget, self).__init__(parent)
            self.setupUi(self)


class AttendanceItemWrapper:
    def __init__(self, attended_list_widget: QListWidget,
                 absented_list_widget: QListWidget,
                 img, name):
        super(AttendanceItemWrapper, self).__init__()
        self.attended_list_widget = attended_list_widget
        self.absented_list_widget = absented_list_widget
        self.prob = 0
        self.frame_num = -1
        self.face_location = None
        self.img = img
        self.matched_face_img = None
        self.name = name
        self.current_item: AttendanceItem = None

    def set_matched_data(self, frame, prob, frame_num, face_location):
        if prob < self.prob:
            return False
        self.prob = prob
        self.frame_num = frame_num
        self.face_location = face_location
        self.matched_face_img = frame
        return True

    def show_on_attend_list(self, threshold_prob):
        if self.current_item is None:
            self.current_item = AttendanceItem(None,
                                               self.img,
                                               None,
                                               None, 0,
                                               self.name)
        if self.prob >= threshold_prob:
            self.current_item.set_matched_data(self.matched_face_img,
                                               self.face_location,
                                               self.prob, self.frame_num)
            self.current_item.add_item(self.attended_list_widget)
        else:
            if self.current_item.list_widget == self.absented_list_widget:
                return
            self.current_item.set_matched_data(None,
                                               None,
                                               0, self.frame_num)
            self.current_item.add_item(self.absented_list_widget)
