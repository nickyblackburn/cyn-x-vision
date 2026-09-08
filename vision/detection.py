from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    source: str

    def to_dict(self):
        x1, y1, x2, y2 = self.bbox

        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "source": self.source,
        }