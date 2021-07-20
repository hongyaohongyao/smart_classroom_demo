from models.pose_estimator import AlphaPoseEstimator
from pipeline_module.core.base_module import BaseModule, TASK_DATA_OK


class AlphaPoseModule(BaseModule):

    def __init__(self, weights, device, skippable=True,
                 face_aligner_weights='weights/mobilenet56_se_external_model_best.torchscript.pth'):
        super(AlphaPoseModule, self).__init__(skippable=skippable)
        self.weights = weights
        self.pose_estimator = AlphaPoseEstimator(weights, device, face_aligner_weights=face_aligner_weights)

    def process_data(self, data):
        preds_kps, preds_scores = self.pose_estimator.estimate(data.frame, data.detections)
        data.keypoints = preds_kps
        data.keypoints_scores = preds_scores
        return TASK_DATA_OK

    def open(self):
        super(AlphaPoseModule, self).open()
        pass
