
import cv2
from colorama import Fore, Style, init

from vision import WeedPenDetector
from vision.camera import Camera
from vision.perception import Perception
from vision.detector import TestDetector
from vision.yolo_detector import YOLODetector


init(autoreset=True)


def log_info(message):
    
    
    
    print(
        f"{Fore.CYAN}[CYN-X-VISION]{Style.RESET_ALL} "
        f"{Fore.GREEN}INFO{Style.RESET_ALL} "
        f"{message}"
    )


def log_success(message):
    print(
        f"{Fore.CYAN}[CYN-X-VISION]{Style.RESET_ALL} "
        f"{Fore.GREEN}✓{Style.RESET_ALL} "
        f"{Fore.WHITE}{message}"
    )


def log_warning(message):
    print(
        f"{Fore.CYAN}[CYN-X-VISION]{Style.RESET_ALL} "
        f"{Fore.YELLOW}⚠{Style.RESET_ALL} "
        f"{message}"
    )


def log_error(message):
    print(
        f"{Fore.CYAN}[CYN-X-VISION]{Style.RESET_ALL} "
        f"{Fore.RED}✗{Style.RESET_ALL} "
        f"{message}"
    )


def main():
    print()
    print(
        f"{Fore.MAGENTA}"
        "╔══════════════════════════════════════╗"
    )
    print(
        f"{Fore.MAGENTA}"
        "║        CYN-X VISION SYSTEM 👁️        ║"
    )
    print(
        f"{Fore.MAGENTA}"
        "╚══════════════════════════════════════╝"
    )
    print()

    log_info("Initializing vision subsystem...")

    camera = Camera()

    perception = Perception()

    yolo = YOLODetector()

    perception.add_detector(yolo)

    perception.add_detector(
    WeedPenDetector.WeedPenDetector()
)

    try:
        log_info("Starting camera...")
        camera.start()

        log_info("Initializing perception system...")
        log_success("Perception system initialized")

        log_info("Loading YOLO detector...")
        log_success("YOLO detector loaded")

        log_success("Vision pipeline: READY")
        print()

        while True:
            frame = camera.read()

            detections = perception.process(frame)

            for detection in detections:
                x1 = detection.bbox[0]
                y1 = detection.bbox[1]
                x2 = detection.bbox[2]
                y2 = detection.bbox[3]

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

            cv2.imshow("CYN-X Vision", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except Exception as error:
        log_error(str(error))

    finally:
        log_info("Shutting down perception system...")
        log_success("Perception system stopped")

        log_info("Shutting down camera...")
        camera.stop()

        cv2.destroyAllWindows()

        log_success("Vision subsystem stopped")


if __name__ == "__main__":
    main()
