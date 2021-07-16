import time
from abc import ABC, abstractmethod

import torch

single_process = True
from queue import Empty

if single_process:
    from queue import Queue
    from threading import Thread, Lock
else:
    from torch.multiprocessing import Queue, Lock
    from torch.multiprocessing import Process as Thread

    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

queueSize = 50

TASK_DATA_OK = 0
TASK_DATA_CLOSE = 1
TASK_DATA_IGNORE = 2
TASK_DATA_SKIP = 3
BALANCE_CEILING_VALUE = 50


class DictData(object):
    def __init__(self):
        pass


class ModuleBalancer:
    def __init__(self):
        self.max_interval = 0
        self.short_stab_interval = self.max_interval
        self.short_stab_module = None
        self.lock = Lock()
        self.ceiling_interval = 0.1

    def get_suitable_interval(self, process_interval, module):
        with self.lock:
            if module == self.short_stab_module:
                self.max_interval = (process_interval + self.max_interval) / 2
                self.short_stab_interval = module.process_interval if module.skippable else self.max_interval
                return 0
            elif process_interval > self.short_stab_interval:
                self.short_stab_module = module
                self.max_interval = process_interval
                self.short_stab_interval = module.process_interval if module.skippable else self.max_interval
                return 0
            else:
                return max(min(self.max_interval - process_interval, self.ceiling_interval), 0)


class TaskData:
    def __init__(self, task_stage, task_flag=TASK_DATA_OK):
        self.data = DictData()
        self.task_stage = task_stage
        self.task_flag = task_flag


class TaskStage:
    def __init__(self):
        self.next_module = None
        self.next_stage = None

    def to_next_stage(self, task_data: TaskData):
        self.next_module().put_task_data(task_data)
        task_data.task_stage = self.next_stage


class BaseModule(ABC):
    def __init__(self, balancer=None, skippable=True):
        self.skippable = skippable
        self.ignore_task_data = TaskData(task_stage=None, task_flag=TASK_DATA_IGNORE)
        self.queue = Queue(maxsize=queueSize)
        self.balancer: ModuleBalancer = balancer
        self.process_interval = 0.01
        self.process_interval_scale = 1
        print(f'created: {self}')

    @abstractmethod
    def process_data(self, data):
        pass

    @abstractmethod
    def open(self):
        self.running = True
        pass

    def _run(self):
        self.running = True
        self.open()
        while self.running:
            task_data = self.product_task_data()
            # 执行条件
            execute_condition = task_data.task_flag == TASK_DATA_OK
            execute_condition = execute_condition or (task_data.task_flag == TASK_DATA_SKIP and not self.skippable)
            # 执行和执行结果
            start_time = time.time()
            execute_result = self.process_data(task_data.data) if execute_condition else task_data.task_flag
            process_interval = min((time.time() - start_time) * self.process_interval_scale, BALANCE_CEILING_VALUE)
            task_stage = task_data.task_stage
            if execute_result != TASK_DATA_SKIP:
                # if str(self).__contains__("FaceEncodingModule"):
                #     print(process_interval, self.queue.qsize())
                self.process_interval = process_interval
            else:
                task_data.task_flag = TASK_DATA_SKIP
            if execute_result == TASK_DATA_IGNORE:
                continue
            else:
                if execute_result == TASK_DATA_CLOSE:
                    task_data.task_flag = TASK_DATA_CLOSE
                    self.close()
                if task_stage.next_stage is not None:
                    task_stage.to_next_stage(task_data)
            if self.balancer is not None:
                suitable_interval = self.balancer.get_suitable_interval(process_interval, self)
                # print( f'process: {process_interval} s,wait: {suitable_interval}s, class: {self},queue: {self.queue.qsize()}')
                if suitable_interval > 0:
                    time.sleep(suitable_interval)

    def start(self):
        p = Thread(target=self._run, args=())
        p.start()
        self.result_worker = p
        return p

    def put_task_data(self, task_data):
        self.queue.put(task_data)
        self._refresh_process_interval_scale()

    def _refresh_process_interval_scale(self):
        self.process_interval_scale = max(self.queue.qsize(), 1)

    def product_task_data(self):
        try:
            task_data = self.queue.get(block=True, timeout=1)
            self._refresh_process_interval_scale()
            return task_data
        except Empty:
            return self.ignore_task_data

    def close(self):
        print(f'closing: {self}')
        self.running = False

    def wait_for_end(self):
        self.result_worker.join()
