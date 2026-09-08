from pathlib import Path
from ultralytics import YOLO

from vision.detection import Detection
from vision.detector import Detector


class WeedPenDetector(Detector):

    def __init__(self, model_path=None, confidence=0.5):

        if model_path is None:
            project_root = Path(__file__).resolve().parent.parent
            model_path = project_root / "models" / "best.pt"

        self.model = YOLO(str(model_path))
        self.confidence = confidence

    @property
    def name(self) -> str:
        return "weed_pen"

    # PUT THE NEW CODE HERE
    def detect(self, frame):
        detections = []

        results = self.model.predict(
            source=frame,
            conf=0.01,
            verbose=False,
        )

        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                print(
                    f"[CYN-X-VISION] YOLO: "
                    f"class={class_id} "
                    f"confidence={confidence:.3f}"
                )

                detections.append(
                    Detection(
                        label="weed_pen",
                        confidence=confidence,
                        bbox=(
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                        ),
                        source="weed_pen_yolo",
                    )
                )

        return detections