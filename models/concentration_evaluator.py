import numpy as np
from numpy import ndarray


class ConcentrationEvaluation(object):
    def __init__(self,
                 primary_levels=None,
                 primary_factor=None,
                 secondary_levels=None):
        self.primary_levels = primary_levels
        self.primary_factor = primary_factor
        self.secondary_levels = secondary_levels
        pass


class ConcentrationEvaluator:
    reclassified_class_actions = [
        0, 0, 2, 1, 1, 1, 1,
        2, 1, 4, 4, 4, 4,
        4, 4, 4, 4,
        3, 3
    ]
    # action_fuzzy_matrix = np.array([
    #     [8.80536914e-01, 1.19167708e-01, 2.95387203e-04, 1.34105589e-08, 1.11512650e-14],  # 正坐
    #     [1.06478877e-01, 7.86778390e-01, 1.06478877e-01, 2.63934722e-04, 1.19826185e-08],  # 抬手
    #     [0.043, 0.1306, 0.2544, 0.3177, 0.2544],  # 放松
    #     [1.11512650e-14, 1.34105589e-08, 2.95387203e-04, 1.19167708e-01, 8.80536914e-01],  # 休息
    #     [0.0047, 0.0574, 0.2571, 0.4238, 0.2571]  # 活动（伸手，转头）
    # ])
    # face_fuzzy_matrix = np.array([
    #     [1.11512650e-14, 1.34105589e-08, 2.95387203e-04, 1.19167708e-01, 8.80536914e-01],  # 严重疲劳
    #     [0.0878, 0.1641, 0.2388, 0.2705, 0.2388],  # 疲劳
    #     [0.2571, 0.4238, 0.2571, 0.0574, 0.0047],  # 非疲劳
    # ])
    # head_pose_fuzzy_matrix = np.array([
    #     [1.91330982e-04, 6.33601192e-03, 7.71884322e-02, 3.45934540e-01, 5.70349634e-01],  # 遮挡
    #     [0.0121, 0.0617, 0.1895, 0.3496, 0.3871],  # 上课低头
    #     [1.0, 0.0, 0.0, 0.0, 0.0],  # 上课抬头
    #     [1.0, 0.0, 0.0, 0.0, 0.0],  # 自习低头
    #     [0.0121, 0.0617, 0.1895, 0.3496, 0.3871],  # 自习抬头
    #     [0.0121, 0.0617, 0.1895, 0.3496, 0.3871],  # 仰头
    # ])
    action_fuzzy_matrix = np.array([
        [1, 0, 0, 0, 0],  # 正坐
        [0, 0.1, 0.8, 0.1, 0],  # 抬手
        [0, 0.5, 0.5, 0, 0],  # 放松
        [0, 0, 0, 0, 1],  # 休息
        [0, 0, 0.3, 0.7, 0]  # 活动（伸手，转头）
    ])
    face_fuzzy_matrix = np.array([
        [0, 0, 0, 0, 1],  # 遮挡
        [1, 0, 0, 0, 0],  # 积极
        [0.7, 0.3, 0, 0, 0],  # 一般
        [0.2, 0.5, 0, 0.1, 0.2],  # 消极
    ])
    head_pose_fuzzy_matrix = np.array([
        [0, 0, 0.1, 0.1, 0.8],  # 遮挡
        [0, 0, 0, 0.2, 0.8],  # 上课低头
        [1.0, 0.0, 0.0, 0.0, 0.0],  # 上课抬头
        [1.0, 0.0, 0.0, 0.0, 0.0],  # 自习低头
        [0, 0, 0.1, 0.5, 0.4],  # 自习抬头
        [0, 0, 0, 0.2, 0.8],  # 仰头
    ])
    evaluation_level = np.array([5, 4, 3, 2, 1])  # 评价等级向量

    def __init__(self, head_pose_split=[-40, 40]):

        self.head_pose_split = head_pose_split
        self.head_pose_section = [  # d1, d2, lbl
            (-200, self.head_pose_split[0], 1),
            (self.head_pose_split[0], self.head_pose_split[1], 2),
            (self.head_pose_split[1], 200, 3)
        ]
        self.on_class = True

    def evaluate(self, action_preds: ndarray,
                 face_preds: ndarray,
                 head_pose_preds: ndarray,
                 face_hidden: ndarray = None) -> ConcentrationEvaluation:
        # 分析二级评价因素
        self.action_preds, self.action_count = self.class_action_reclassify(action_preds)
        self.face_preds, self.face_count = self.face_action_reclassify(face_preds, face_hidden)
        self.head_pose_preds, self.head_pose_count = self.head_pose_reclassify(head_pose_preds, face_hidden)
        # 二级评价等级
        self.action_levels = self.action_preds @ self.action_fuzzy_matrix @ self.evaluation_level
        self.face_levels = self.face_preds @ self.face_fuzzy_matrix @ self.evaluation_level
        self.head_pose_levels = self.head_pose_preds @ self.head_pose_fuzzy_matrix @ self.evaluation_level

        self.secondary_levels = np.hstack([
            self.action_levels[..., np.newaxis],
            self.face_levels[..., np.newaxis],
            self.head_pose_levels[..., np.newaxis]
        ])

        # 分析一级评级因素
        self.action_info_entropy = self.info_entropy(self.action_count / np.sum(self.action_count))
        self.face_info_entropy = self.info_entropy(self.face_count / np.sum(self.face_count))
        self.head_pose_info_entropy = self.info_entropy(self.head_pose_count / np.sum(self.head_pose_count))

        self.primary_factor = self.softmax(np.array([self.action_info_entropy,
                                                     self.face_info_entropy,
                                                     self.head_pose_info_entropy]))
        # 一级评价因素等级
        self.primary_levels = self.secondary_levels @ self.primary_factor
        return ConcentrationEvaluation(self.primary_levels,
                                       self.primary_factor,
                                       self.secondary_levels)

    def class_action_reclassify(self, action_preds):
        """
        重新分类课堂动作标签
        """
        reclassified_preds = np.empty_like(action_preds)
        for lbl, new_class in enumerate(self.reclassified_class_actions):
            reclassified_preds[action_preds == lbl] = new_class
        min_len = self.action_fuzzy_matrix.shape[0]
        result = np.eye(min_len)[reclassified_preds]
        count_vec = np.bincount(reclassified_preds, minlength=min_len)
        return result, count_vec

    def face_action_reclassify(self, face_preds, face_hidden=None):
        """
        重新分类面部疲劳标签
        0  "nature" 1   "happy" 2   "confused" 3 "amazing"
        """
        result = np.empty_like(face_preds)
        result[(face_preds == 1) | (face_preds == 3)] = 1
        result[face_preds == 0] = 2
        result[face_preds == 2] = 3

        if face_hidden is not None:
            result[face_hidden] = 0
        min_len = self.face_fuzzy_matrix.shape[0]
        count_vec = np.bincount(result, minlength=min_len)
        return np.eye(min_len)[result], count_vec

    def head_pose_reclassify(self, head_pose_preds, face_hidden=None):
        """
        离散化分类头部角度
        """
        print(head_pose_preds.flatten().tolist())
        discretization_head_pose = np.empty_like(head_pose_preds, dtype=np.int64)
        for d1, d2, lbl in self.head_pose_section:
            discretization_head_pose[(d1 < head_pose_preds) & (head_pose_preds <= d2)] = lbl
        if face_hidden is not None:
            discretization_head_pose[face_hidden] = 0

        # 分解上课和自习状态
        count_ = np.array([np.count_nonzero(discretization_head_pose == 1),
                           np.count_nonzero(discretization_head_pose == 2)])
        sum_count_ = np.sum(count_)
        count_ = np.array([0, 1]) if sum_count_ == 0 else count_ / sum_count_
        # count_ = np.array([0, 1]) if sum_count_ == 0 else np.round(count_ / sum_count_)

        encode = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, count_[1], 0, count_[0], 0, 0],
            [0, 0, count_[1], 0, count_[0], 0],
            [0, 0, 0, 0, 0, 1]
        ])
        discretization_head_pose = discretization_head_pose.flatten()
        result = encode[discretization_head_pose]
        min_len = self.head_pose_fuzzy_matrix.shape[0]
        count_vec = np.bincount(discretization_head_pose, minlength=min_len)

        return result, count_vec

    @staticmethod
    def info_entropy(y: ndarray):
        delta = 1e-16  # 添加一个微小值可以防止负无限大(np.log(0))的发生。
        n = y.shape[0]
        info_entropy = (y @ np.log(y + delta) - np.log(1 / n)) / np.log(n)
        return info_entropy.clip(0, 1)

    @staticmethod
    def softmax(x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    def get_expressions(self, face_landmarks):
        return np.array([self.get_expression(marks) for marks in face_landmarks])

    @staticmethod
    def get_expression(marks):
        """
        通过关键点获取表情
        :param marks: 关键点。格式为<br />
        [[1,1] <br />
        [2,2]] <br />
        共 68 个点
        :return: 0  "nature" 1   "happy" 2   "confused" 3 "amazing"
        """
        # 脸的宽度
        face_width = marks[14][0] - marks[0][0]

        # 嘴巴张开程度
        mouth_higth = (marks[66][1] - marks[62][1]) / face_width

        # 通过两个眉毛上的10个特征点，分析挑眉程度和皱眉程度
        brow_sum = 0  # 高度之和
        frown_sum = 0  # 两边眉毛距离之和

        # 眼睛睁开程度
        eye_sum = (marks[41][1] - marks[37][1] + marks[40][1] - marks[38][1] +
                   marks[47][1] - marks[43][1] + marks[46][1] - marks[44][1])
        eye_hight = (eye_sum / 4) / face_width
        # print("眼睛睁开距离与识别框高度之比：",round(eye_open/self.face_width,3))

        # 头部倾斜程度
        slope = (marks[42][1] - marks[39][1]) / (marks[42][0] - marks[39][0])

        # 两嘴角中间位置占据上下唇高度比例
        center_point = (marks[54][1] + marks[48][1]) / 2
        min_month_point = min([marks[56][1], marks[57][1], marks[58][1]])
        max_month_point = min([marks[50][1], marks[51][1], marks[52][1]])
        mouth_corner_proportion = (center_point - min_month_point) / (max_month_point - min_month_point)

        # 分情况讨论
        # 张嘴，可能是开心或者惊讶
        if mouth_higth >= 0.04:
            if mouth_corner_proportion >= 0.55:
                return 1  # "happy"
            else:
                return 3  # "amazing"

        # 没有张嘴，可能是正常和疑惑
        else:
            if abs(slope) >= 0.3:
                return 2  # "confused"
            else:
                return 0  # "nature"


if __name__ == '__main__':
    # # 设置类别的数量
    # num_classes = 10
    # # 需要转换的整数
    # arr = [1, 3, 4, 5, 9]
    # # 将整数转为一个10位的one hot编码
    # print(np.eye(10)[arr])

    # np.tile(np.array([[*range(5)]]), (head_pose_preds.shape[0], 1))
    # print(ConcentrationEvaluator.info_entropy(np.array([0.5, 0.5])))
    # preds = np.array([
    #     [1, 0, 0],
    #     [0, 0, 1],
    #     [0, 1, 0],
    #     [0, 0.5, 0.5]
    # ])
    #
    # level = np.array([3, 2, 1])
    # print(preds @ level)

    # a = np.array([1, 2, 3, 4, 5])[..., np.newaxis]
    # b = np.array([5, 4, 3, 2, 1])[..., np.newaxis]
    # c = np.array([1, 1, 1, 1, 1])[..., np.newaxis]
    # print(np.hstack([a, b, c]))
    # print(np.array([
    #     [0, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 1, 0]
    # ]) @ np.array([
    #     [0, 0, 0.1, 0.4, 0.5],  # 遮挡
    #     [0, 0, 0, 0.2, 0.8],  # 上课低头
    #     [1.0, 0.0, 0.0, 0.0, 0.0],  # 上课抬头
    #     [1.0, 0.0, 0.0, 0.0, 0.0],  # 自习低头
    #     [0, 0, 0.1, 0.5, 0.4],  # 自习抬头
    #     [0, 0, 0, 0.2, 0.8]  # 仰头
    # ]) @ np.array([5, 4, 3, 2, 1]))
    print(np.mean(np.array(
        [
            [1, 1, 1],
            [2, 2, 2, ],
            [3, 3, 3],

        ]
    ), axis=0))
    pass
