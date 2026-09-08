import cv2


class Camera:
    def __init__(self, device=0, width=1280, height=720):
        self.device = device
        self.width = width
        self.height = height
        self.capture = None

    def start(self):
        self.capture = cv2.VideoCapture(self.device)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Unable to open camera device {self.device}"
            )

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        if self.capture is None:
            raise RuntimeError("Camera has not been started.")

        success, frame = self.capture.read()

        if not success:
            raise RuntimeError("Failed to read frame from camera.")

        return frame

    def stop(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None