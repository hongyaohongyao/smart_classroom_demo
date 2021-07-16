import torch

from models.action_analysis import CheatingActionAnalysis
from models.classroom_action_classifier import ClassroomActionClassifier
from models.pose_estimator import PnPPoseEstimator
from pipeline_module.core.base_module import TASK_DATA_OK, BaseModule


class CheatingActionModule(BaseModule):
    raw_class_names = ["seat", "write", "stretch", "hand_up_R", "hand_up_L",
                       "hand_up_highly_R", "hand_up_highly_L",
                       "relax", "hand_up", "pass_R", "pass_L", "pass2_R", "pass2_L",
                       "turn_round_R", "turn_round_L", "turn_head_R", "turn_head_L",
                       "sleep", "lower_head"]
    class_names = ["正常", "传纸条", "低头偷看", "东张西望"]
    use_keypoints = [x for x in range(11)] + [17, 18, 19]
    class_of_passing = [9, 10, 11, 12]
    class_of_peep = [18]
    class_of_gazing_around = [13, 14, 15, 16]

    # 0 正常坐姿不动
    # 1 正常低头写字
    # 2 正常伸懒腰
    # 3 举右手低
    # 4 举左手低
    # 5 举右手高
    # 6 举左手高
    # 7 起立
    # 8 抬手
    # 9 右伸手
    # 10 左伸手
    # 11 右伸手2
    # 12 左伸手2
    # 13 右转身
    # 14 左转身
    # 15 右转头
    # 16 左转头
    # 17 上课睡觉
    # 18 严重低头

    def __init__(self, weights, device='cpu', img_size=(480, 640), skippable=True):
        super(CheatingActionModule, self).__init__(skippable=skippable)
        self.weights = weights
        self.classifier = ClassroomActionClassifier(weights, device)
        self.pnp = PnPPoseEstimator(img_size=img_size)

    def process_data(self, data):
        data.num_of_cheating = 0
        data.num_of_normal = 0
        data.num_of_passing = 0
        data.num_of_peep = 0
        data.num_of_gazing_around = 0
        if data.detections.shape[0] > 0:
            # 行为识别
            data.classes_probs = self.classifier.classify(data.keypoints[:, self.use_keypoints])
            # 最佳行为分类
            data.raw_best_preds = torch.argmax(data.classes_probs, dim=1)
            data.best_preds = [self.reclassify(idx) for idx in data.raw_best_preds]
            data.raw_classes_names = self.raw_class_names
            data.classes_names = self.class_names
            # 头背部姿态估计
            data.head_pose = [self.pnp.solve_pose(kp) for kp in data.keypoints[:, 26:94, :2].numpy()]
            data.draw_axis = self.pnp.draw_axis
            data.head_pose_euler = [self.pnp.get_euler(*vec) for vec in data.head_pose]
            # 传递动作识别
            is_passing_list = CheatingActionAnalysis.is_passing(data.keypoints)
            # 头部姿态辅助判断转头
            for i in range(len(data.best_preds)):
                if data.best_preds[i] == 0:
                    if is_passing_list[i] != 0:
                        data.best_preds[i] = 1
                    elif data.head_pose_euler[i][1][0] < -50:
                        data.best_preds[i] = 2
            data.pred_class_names = [self.class_names[i] for i in data.best_preds]
            # 统计人数
            data.num_of_normal = data.best_preds.count(0)
            data.num_of_passing = data.best_preds.count(1)
            data.num_of_peep = data.best_preds.count(2)
            data.num_of_gazing_around = data.best_preds.count(3)
            data.num_of_cheating = data.detections.shape[0] - data.num_of_normal

        return TASK_DATA_OK

    def reclassify(self, class_idx):
        if class_idx in self.class_of_passing:
            return 1
        elif class_idx in self.class_of_peep:
            return 2
        elif class_idx in self.class_of_gazing_around:
            return 3
        else:
            return 0

    def open(self):
        super(CheatingActionModule, self).open()
        pass
