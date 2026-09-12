from __future__ import annotations

import cv2
from pathlib import Path


# ============================================================
# CYN-X VISION - MANUAL YOLO LABELER
# ============================================================

IMAGE_DIR = Path("dataset/organized")
LABEL_DIR = Path("dataset/labels")

CLASS_ID = 0
CLASS_NAME = "weed_pen"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ------------------------------------------------------------
# Global mouse state
# ------------------------------------------------------------

drawing = False
start_x = 0
start_y = 0
current_x = 0
current_y = 0

boxes = []


# ------------------------------------------------------------
# Mouse callback
# ------------------------------------------------------------

def mouse_callback(event, x, y, flags, param):
    global drawing
    global start_x, start_y
    global current_x, current_y
    global boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

        start_x = x
        start_y = y
        current_x = x
        current_y = y

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:
            current_x = x
            current_y = y

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        current_x = x
        current_y = y

        x1 = min(start_x, current_x)
        y1 = min(start_y, current_y)
        x2 = max(start_x, current_x)
        y2 = max(start_y, current_y)

        # Ignore tiny accidental clicks
        if (x2 - x1) >= 5 and (y2 - y1) >= 5:
            boxes.append((x1, y1, x2, y2))


# ------------------------------------------------------------
# Convert pixel box → YOLO normalized coordinates
# ------------------------------------------------------------

def convert_to_yolo(box, image_width, image_height):

    x1, y1, x2, y2 = box

    center_x = ((x1 + x2) / 2) / image_width
    center_y = ((y1 + y2) / 2) / image_height

    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    return (
        center_x,
        center_y,
        width,
        height,
    )


# ------------------------------------------------------------
# Save YOLO label
# ------------------------------------------------------------

def save_label(image_path, image_width, image_height):

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    with open(label_path, "w", encoding="utf-8") as file:

        for box in boxes:

            x, y, w, h = convert_to_yolo(
                box,
                image_width,
                image_height,
            )

            file.write(
                f"{CLASS_ID} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{w:.6f} "
                f"{h:.6f}\n"
            )

    print(
        f"[SAVED] {label_path} "
        f"({len(boxes)} box(es))"
    )


# ------------------------------------------------------------
# Draw UI
# ------------------------------------------------------------

def draw_interface(image, image_index, total, image_path):

    display = image.copy()

    # Existing boxes
    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = box

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            f"{CLASS_NAME} #{i + 1}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Current mouse rectangle
    if drawing:

        x1 = min(start_x, current_x)
        y1 = min(start_y, current_y)
        x2 = max(start_x, current_x)
        y2 = max(start_y, current_y)

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

    # Header
    cv2.rectangle(
        display,
        (0, 0),
        (display.shape[1], 45),
        (30, 30, 30),
        -1,
    )

    header = (
        f"{image_index + 1}/{total}  "
        f"{image_path.name}  "
        f"Boxes: {len(boxes)}"
    )

    cv2.putText(
        display,
        header,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    # Footer
    footer_y = display.shape[0] - 10

    cv2.putText(
        display,
        "ENTER=save  N=next  B=back  R=reset  Q=quit",
        (10, footer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    return display


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    global boxes

    LABEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:

        print(
            f"[ERROR] No images found in {IMAGE_DIR}"
        )

        return

    print("=" * 60)
    print("          CYN-X VISION MANUAL LABELER")
    print("=" * 60)
    print()
    print(f"Images: {len(images)}")
    print(f"Class:  {CLASS_ID} = {CLASS_NAME}")
    print()
    print("CONTROLS")
    print("------------------------------------------")
    print("Mouse drag  = create bounding box")
    print("ENTER       = save and next")
    print("N           = next")
    print("B           = previous")
    print("R           = remove all boxes")
    print("Q           = quit")
    print()

    index = 0

    cv2.namedWindow(
        "CYN-X Vision Labeler",
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        "CYN-X Vision Labeler",
        mouse_callback,
    )

    while 0 <= index < len(images):

        image_path = images[index]

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"[ERROR] Could not read {image_path}"
            )

            index += 1
            continue

        height, width = image.shape[:2]

        # Start fresh for this image
        boxes = []

        # If a label already exists, load it
        existing_label = LABEL_DIR / f"{image_path.stem}.txt"

        if existing_label.exists():

            try:

                with open(
                    existing_label,
                    "r",
                    encoding="utf-8",
                ) as file:

                    for line in file:

                        parts = line.strip().split()

                        if len(parts) != 5:
                            continue

                        class_id = int(parts[0])

                        if class_id != CLASS_ID:
                            continue

                        xc = float(parts[1]) * width
                        yc = float(parts[2]) * height
                        w = float(parts[3]) * width
                        h = float(parts[4]) * height

                        x1 = int(xc - w / 2)
                        y1 = int(yc - h / 2)
                        x2 = int(xc + w / 2)
                        y2 = int(yc + h / 2)

                        boxes.append(
                            (x1, y1, x2, y2)
                        )

            except Exception:
                boxes = []

        while True:

            display = draw_interface(
                image,
                index,
                len(images),
                image_path,
            )

            cv2.imshow(
                "CYN-X Vision Labeler",
                display,
            )

            key = cv2.waitKey(30) & 0xFF

            # ENTER = save
            if key in (13, 10):

                if not boxes:

                    print(
                        f"[SKIP] {image_path.name} "
                        "has no boxes."
                    )

                    index += 1
                    break

                save_label(
                    image_path,
                    width,
                    height,
                )

                index += 1
                break

            # N = next without saving
            elif key in (ord("n"), ord("N")):

                print(
                    f"[NEXT] {image_path.name}"
                )

                index += 1
                break

            # B = previous
            elif key in (ord("b"), ord("B")):

                if index > 0:
                    index -= 1

                break

            # R = reset boxes
            elif key in (ord("r"), ord("R")):

                boxes = []

                print(
                    f"[RESET] {image_path.name}"
                )

            # Q = quit
            elif key in (ord("q"), ord("Q")):

                print()
                print("[EXIT] Labeler stopped.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("                 COMPLETE")
    print("=" * 60)
    print()
    print(f"Labels saved to: {LABEL_DIR}")


if __name__ == "__main__":
    main()