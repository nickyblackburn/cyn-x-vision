from abc import ABC, abstractmethod

from vision.detection import Detection


class Detector(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def detect(self, frame) -> list[Detection]:
        pass


class TestDetector(Detector):

    @property
    def name(self):
        return "test"

    def detect(self, frame) -> list[Detection]:
        return []