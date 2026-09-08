from ultralytics import YOLO

from vision.detector import Detector
from vision.detection import Detection


class YOLODetector(Detector):

    def __init__(self, model="yolo11n.pt", confidence=0.40):
        self.model_path = model
        self.confidence = confidence
        self.model = YOLO(model)

    @property
    def name(self):
        return "YOLO"

    def detect(self, frame) -> list[Detection]:
        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                label = self.model.names[class_id]

                detection = Detection(
                    label=label,
                    confidence=confidence,
                    bbox=(
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ),
                    source=self.name,
                )

                detections.append(detection)

        return detections