from typing import Union

import numpy
from asm.api.base import ModuleTask, ModuleTaskInput, ModuleTaskOutput, ModuleConfiguration, ModuleInformation
from asm.api.cv import ASMDetector


class C7(ASMDetector):
    def process(self, frame: numpy.ndarray) -> tuple[bool, float, float, float, float]:
        pass

    def module_info(self) -> ModuleInformation:
        pass

    def configuration(self, configuration: ModuleConfiguration):
        pass

    def task(self, task: ModuleTask, task_input: ModuleTaskInput) -> Union[ModuleTaskOutput, None]:
        pass