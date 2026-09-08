
import cv2
from colorama import Fore, Style, init

from vision.camera import Camera
from vision.perception import Perception
from vision.detector import TestDetector


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
    perception.add_detector(TestDetector())

    try:
        log_info("Starting camera...")
        camera.start()

        log_success("Camera initialized")
        log_info("Resolution: 1280x720")

        log_info("Initializing perception system...")
        log_success("Perception system initialized")

        log_info("Loading detectors...")
        log_success("Test detector loaded")

        log_success("Vision pipeline: READY")
        log_info("Press Q to exit")
        print()

        while True:
            frame = camera.read()

            detections = perception.process(frame)

            for detection in detections:
                log_info(
                    f"{detection.label} "
                    f"({detection.confidence:.2f}) "
                    f"[{detection.source}]"
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
