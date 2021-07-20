import sys

import qdarkstyle

from smart_classroom.cheating_detection_app import CheatingDetectionApp
from smart_classroom.class_concentration_app import ClassConcentrationApp
from smart_classroom.dynamic_attendance_app import DynamicAttendanceApp
from smart_classroom.face_register_app import FaceRegisterApp

try:
    import smart_classroom_rc
except ImportError:
    pass
import torch
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication

from ui.smart_classroom import Ui_MainWindow as SmartClassroomMainWindow

torch._C._jit_set_profiling_mode(False)
torch.jit.optimized_execution(False)


class SmartClassroomApp(QMainWindow, SmartClassroomMainWindow):

    def __init__(self, parent=None):
        super(SmartClassroomApp, self).__init__(parent)
        self.setupUi(self)
        self.cheating_detection_widget = CheatingDetectionApp()
        self.cheating_detection_widget.setObjectName("cheating_detection_widget")
        self.tabWidget.addTab(self.cheating_detection_widget, "作弊检测")

        self.class_concentration_widget = ClassConcentrationApp()
        self.class_concentration_widget.setObjectName("class_concentration_widget")
        self.tabWidget.addTab(self.class_concentration_widget, "课堂专注度分析")

        self.face_register_widget = FaceRegisterApp()
        self.face_register_widget.setObjectName("face_register_widget")
        self.tabWidget.addTab(self.face_register_widget, "人脸注册")

        self.dynamic_attendance_widget = DynamicAttendanceApp()
        self.dynamic_attendance_widget.setObjectName("dynamic_attendance_widget")
        self.tabWidget.addTab(self.dynamic_attendance_widget, "动态点名")

        self.current_tab_widget = self.tabWidget.currentWidget()

        def current_tab_change(idx, self=self):
            if self.current_tab_widget is not None:
                self.current_tab_widget.close()
            self.current_tab_widget = self.tabWidget.widget(idx)
            self.current_tab_widget.open()

        self.tabWidget.currentChanged.connect(current_tab_change)
        # def change_tab_widget(index):
        #     self.tabWidget.widget(index).close()
        #
        # self.tabWidget.currentChanged.connect()
        # _translate = QtCore.QCoreApplication.translate
        # self.tabWidget.setTabText(self.tabWidget.indexOf(self.cheating_detection_widget),
        #                           _translate("MainWindow", "作弊检测"))

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.cheating_detection_widget.close()
        self.face_register_widget.close()
        self.dynamic_attendance_widget.close()
        super(SmartClassroomApp, self).closeEvent(a0)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # from QcureUi import cure
    #
    # window = cure.Windows(SmartClassroomApp(), 'trayname', True, title='智慧教室')
    window = SmartClassroomApp()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    # run
    window.show()
    sys.exit(app.exec_())
