import time

import cv2

from pipeline_module.core.base_module import BaseModule, TASK_DATA_CLOSE, TASK_DATA_OK, TaskData, TASK_DATA_SKIP, \
    TASK_DATA_IGNORE


class VideoModule(BaseModule):

    def __init__(self, source=0, fps=25, skippable=False):
        super(VideoModule, self).__init__(skippable=skippable)
        self.task_stage = None
        self.source = source
        self.cap = None
        self.frame = None
        self.ret = False
        self.skip_timer = 0
        self.set_fps(fps)
        self.loop = True

    def process_data(self, data):
        if not self.ret:
            if self.loop:
                self.open()
                return TASK_DATA_IGNORE
            else:
                return TASK_DATA_CLOSE
        data.source_fps = self.fps
        data.frame = self.frame
        self.ret, self.frame = self.cap.read()
        result = TASK_DATA_OK
        if self.skip_timer != 0:
            result = TASK_DATA_SKIP
            data.skipped = None
        skip_gap = int(self.fps * self.balancer.short_stab_interval)
        # skip_gap = 1
        # print(self.balancer.short_stab_module, self.balancer.short_stab_interval)
        if self.skip_timer > skip_gap:
            self.skip_timer = 0
        else:
            self.skip_timer += 1
        time.sleep(self.interval)
        return result

    def product_task_data(self):
        return TaskData(self.task_stage)

    def set_fps(self, fps):
        self.fps = fps
        self.interval = 1 / fps

    def open(self):
        super(VideoModule, self).open()
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.source)
        if self.cap.isOpened():
            self.set_fps(self.cap.get(cv2.CAP_PROP_FPS))
            self.ret, self.frame = self.cap.read()
            print("视频源帧率: ", self.fps)
        pass
