from vision.detector import Detector


class Perception:

    def __init__(self):
        self.detectors: list[Detector] = []

    def add_detector(self, detector: Detector):
        self.detectors.append(detector)

    def process(self, frame):
        detections = []

        for detector in self.detectors:
            results = detector.detect(frame)
            detections.extend(results)

        return detections