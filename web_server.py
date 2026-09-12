import cv2
import threading
import time

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from vision.camera import Camera
from vision.perception import Perception
from vision.yolo_detector import YOLODetector


app = FastAPI(title="CYN-X Vision")


camera = Camera()
perception = Perception()

yolo = YOLODetector()
perception.add_detector(yolo)


latest_frame = None
frame_lock = threading.Lock()

running = False
fps = 0.0


def vision_loop():
    global latest_frame, running, fps

    frame_count = 0
    start_time = time.time()

    while running:
        frame = camera.read()

        detections = perception.process(frame)

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 255),
                2,
            )

            label = (
                f"{detection.label} "
                f"{detection.confidence:.0%}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )

        frame_count += 1

        elapsed = time.time() - start_time

        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        with frame_lock:
            latest_frame = frame.copy()


def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.01)
            continue

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )


@app.on_event("startup")
def startup():
    global running

    camera.start()

    running = True

    thread = threading.Thread(
        target=vision_loop,
        daemon=True,
    )

    thread.start()


@app.on_event("shutdown")
def shutdown():
    global running

    running = False

    camera.stop()


@app.get("/", response_class=HTMLResponse)
def index():
    with open("web/index.html", "r", encoding="utf-8") as file:
        return file.read()


@app.get("/video")
def video():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/status")
def status():
    return {
        "running": running,
        "fps": round(fps, 1),
    }