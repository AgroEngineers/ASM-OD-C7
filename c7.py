from typing import Union

import cv2
import numpy
from asm.api.base import ModuleTask, ModuleTaskInput, ModuleTaskOutput, ModuleConfiguration, ModuleInformation
from asm.api.cv import ASMDetector, DetectedObject


class C7(ASMDetector):
    async def process(self, frame: numpy.ndarray) -> Union[DetectedObject, None]:
        h, w = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = numpy.array([0, 120, 70])
        upper_red1 = numpy.array([10, 255, 255])
        lower_red2 = numpy.array([170, 120, 70])
        upper_red2 = numpy.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask = mask1 | mask2

        kernel = numpy.ones((5, 5), numpy.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        ys, xs = numpy.where(mask > 0)
        if len(xs) == 0:
            return None

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        if xmin == 0 or ymin == 0 or xmax >= w - 1 or ymax >= h - 1:
            return None

        return DetectedObject(True, xmin, xmax, ymin, ymax)

    def module_info(self) -> ModuleInformation:
        return ModuleInformation(
            name="Detector-C7",
            version="1.0.0"
        )

    def configuration(self, configuration: ModuleConfiguration):
        return None

    def task(self, task: ModuleTask, task_input: ModuleTaskInput) -> Union[ModuleTaskOutput, None]:
        return None
