from __future__ import annotations

import json
import time

import zmq


class VisionPublisher:
    """
    Publishes CYN-X Vision perception data over ZeroMQ.
    """

    def __init__(self, address: str = "tcp://127.0.0.1:5555"):
        self.address = address

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)

        self.socket.bind(self.address)

        print(f"[ZMQ] Vision publisher listening on {self.address}")

    def publish_detections(
        self,
        detections: list[dict],
        frame_id: int,
    ) -> None:

        message = {
            "type": "vision.detections",
            "timestamp": time.time(),
            "frame_id": frame_id,
            "objects": detections,
        }

        self.socket.send_json(message)

    def close(self) -> None:
        self.socket.close()
        self.context.term()