import numpy as np

from face_recog.models import face_recog
from face_recog.models.face_boxes_location import FaceBoxesLocation
from models.pose_estimator import PnPPoseEstimator
from pipeline_module.core.base_module import BaseModule, TASK_DATA_OK


class FaceDetectionModule(BaseModule):

    def __init__(self, skippable=True):
        super(FaceDetectionModule, self).__init__(skippable=skippable)
        self.fbl = FaceBoxesLocation()
        self.pnp = PnPPoseEstimator()

    def process_data(self, data):
        frame = data.frame
        # 人脸检查
        data.face_locations = self.fbl.face_location(frame)
        # 人脸对齐
        data.face_landmarks = np.array(face_recog.face_landmarks(frame, data.face_locations),
                                       dtype=np.float)
        # 头部姿态估计
        data.head_pose = [self.pnp.solve_pose(kp) for kp in data.face_landmarks]
        data.draw_axis = self.pnp.draw_axis
        data.head_pose_euler = [self.pnp.get_euler(*vec) for vec in data.head_pose]
        return TASK_DATA_OK

    def open(self):
        super(FaceDetectionModule, self).open()
        pass
