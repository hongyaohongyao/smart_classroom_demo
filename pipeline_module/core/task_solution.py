from pipeline_module.core.base_module import BaseModule, TaskStage, ModuleBalancer


class TaskSolution:

    def __init__(self):
        self.modules = []
        self.start_stage: TaskStage = TaskStage()
        self.current_stage: TaskStage = self.start_stage
        self.source_module = None
        self.balancer = ModuleBalancer()

    def set_source_module(self, source_module):
        source_module.balancer = self.balancer
        self.source_module = source_module
        source_module.task_stage = self.start_stage
        return self

    def set_next_module(self, next_module: BaseModule):
        next_module.balancer = self.balancer
        next_stage = TaskStage()
        self.current_stage.next_module = lambda: next_module
        self.current_stage.next_stage = next_stage
        self.current_stage = next_stage
        self.modules.append(next_module)
        return self

    def start(self):
        for module in self.modules:
            print(f'starting modules {module}')
            module.start()
        self.source_module.start()

    def wait_for_end(self):
        self.source_module.wait_for_end()
        for module in self.modules:
            module.wait_for_end()

    def close(self):
        self.source_module.close()
        for module in self.modules:
            print(f'closing modules {module}')
            module.close()
