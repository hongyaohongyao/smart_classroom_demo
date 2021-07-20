# -*- coding: utf-8 -*-

import numpy as np

# ==============================================================================
#   landmarks格式转换函数
#       输入：dlib格式的landmarks
#       输出：numpy格式的landmarks
# ==============================================================================       2


"""
首先实例化这个detector类，判断睡眠需要临近30帧的判断结果，有一半以上是睡眠那这个人就是睡觉
输入 landmarks_list : [[landmarks1] [landmarks2] ... [landmarksN]]
输出 sleep_states : [0 0 0 1 ... 1] 0是正常 1是犯困
"""


class SleepDetector(object):

    # TODO 检测器V2 如果v1不能用，可以在你那边手动做一个queue，在外部统计相邻30帧的睡觉情况，然后在做判断
    # 根据某一帧来判断是否犯困
    @staticmethod
    def detector_v2(landmarks_list):
        sleep_states = np.zeros_like(landmarks_list).tolist()
        for idx, landmarks in enumerate(landmarks_list):
            # 计算欧氏距离
            d1 = np.linalg.norm(landmarks[37] - landmarks[41])
            d2 = np.linalg.norm(landmarks[38] - landmarks[40])
            d3 = np.linalg.norm(landmarks[43] - landmarks[47])
            d4 = np.linalg.norm(landmarks[44] - landmarks[46])
            d_mean = (d1 + d2 + d3 + d4) / 4
            d5 = np.linalg.norm(landmarks[36] - landmarks[39])
            d6 = np.linalg.norm(landmarks[42] - landmarks[45])
            d_reference = (d5 + d6) / 2
            d_judge = d_mean / d_reference
            # print(d_judge)
            flag = int(d_judge < 0.18)  # 睁/闭眼判定标志:根据阈值判断是否闭眼,闭眼flag=1,睁眼flag=0 (阈值可调)
            sleep_states[idx] = flag

            # # flag入队
            # self.queue[idx] = self.queue[idx][1:len(self.queue[idx])] + [flag]
            #
            # # 判断是否疲劳：根据时间序列中低于阈值的元素个数是否超过一半
            # if sum(self.queue[idx]) > len(self.queue[idx]) / 2:
            #     self.sleep_states[idx] = 1
            # else:
            #     self.sleep_states[idx] = 0
        return sleep_states
