from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# HELPERS
# ============================================================

def is_negative_image(image_path: Path) -> bool:
    """
    Return True when the filename identifies a known
    negative training image.

    Examples:

        negative_00001.jpg
        negative_00002.jpg
        negative-00003.jpg

    Negative images NEVER go through YOLO.
    """

    name = image_path.stem.lower()

    return name.startswith(
        (
            "negative_",
            "negative-",
        )
    )


def clear_directory(directory: Path) -> None:
    """
    Completely remove a directory and recreate it.

    This is intentionally used only for generated
    classifications/review output.
    """

    if directory.exists():

        print(
            f"[CLEANUP] Clearing: {directory}"
        )

        shutil.rmtree(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def yolo_box(
    x1,
    y1,
    x2,
    y2,
    img_w,
    img_h,
):
    """
    Convert pixel coordinates into normalized
    YOLO format.
    """

    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h

    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h

    return (
        x_center,
        y_center,
        width,
        height,
    )


def create_empty_label(
    label_path: Path,
) -> None:
    """
    Create an empty YOLO label file.

    Empty label means:

        This image contains ZERO weed_pen objects.
    """

    label_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_path.write_text(
        "",
        encoding="utf-8",
    )


def copy_image(
    source_path: Path,
    destination_path: Path,
) -> None:
    """
    Copy an image while normalizing it to RGB JPEG.
    """

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with Image.open(
        source_path
    ) as image:

        image = image.convert(
            "RGB"
        )

        image.save(
            destination_path,
            quality=95,
        )


def draw_boxes(
    image_path: Path,
    detections,
    output_path: Path,
) -> None:
    """
    Create a visual review image with predicted boxes.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with Image.open(
        image_path
    ) as source:

        image = source.convert(
            "RGB"
        )

    draw = ImageDraw.Draw(
        image
    )

    for detection in detections:

        (
            x1,
            y1,
            x2,
            y2,
            confidence,
        ) = detection

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            outline="red",
            width=4,
        )

        label = (
            f"weed_pen "
            f"{confidence:.2f}"
        )

        text_top = max(
            0,
            y1 - 25,
        )

        draw.rectangle(
            [
                x1,
                text_top,
                x1 + 150,
                y1,
            ],
            fill="red",
        )

        draw.text(
            (
                x1 + 5,
                max(
                    0,
                    y1 - 22,
                ),
            ),
            label,
            fill="white",
        )

    image.save(
        output_path,
        quality=95,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "CYN-X Vision YOLO automatic "
            "labeling tool."
        )
    )

    parser.add_argument(
        "--model",
        default=(
            "runs/detect/"
            "runs/weed_pen_bootstrap/"
            "weights/best.pt"
        ),
        help="YOLO model to use.",
    )

    parser.add_argument(
        "--images",
        default="dataset/organized",
        help="Directory containing organized images.",
    )

    parser.add_argument(
        "--labels",
        default="dataset/labels",
        help="Directory for YOLO labels.",
    )

    parser.add_argument(
        "--preview",
        default="dataset/review",
        help="Directory for positive review images.",
    )

    parser.add_argument(
        "--negative-review",
        default="dataset/negative_review",
        help=(
            "Directory for positive images "
            "where YOLO finds nothing."
        ),
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.70,
        help=(
            "Confidence required for "
            "automatic labeling."
        ),
    )

    parser.add_argument(
        "--review-confidence",
        type=float,
        default=0.50,
        help=(
            "Minimum confidence required "
            "to send a detection to review."
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size.",
    )

    args = parser.parse_args()

    # ========================================================
    # PATHS
    # ========================================================

    model_path = Path(
        args.model
    )

    image_dir = Path(
        args.images
    )

    label_dir = Path(
        args.labels
    )

    preview_dir = Path(
        args.preview
    )

    negative_review_dir = Path(
        args.negative_review
    )

    # ========================================================
    # CHECK REQUIRED PATHS
    # ========================================================

    if not model_path.exists():

        print(
            f"[ERROR] Model not found:"
        )

        print(
            f"        {model_path}"
        )

        return

    if not image_dir.exists():

        print(
            f"[ERROR] Image directory not found:"
        )

        print(
            f"        {image_dir}"
        )

        return

    # ========================================================
    # STARTUP
    # ========================================================

    print()
    print("=" * 70)
    print(
        "              CYN-X VISION AUTO LABELER"
    )
    print("=" * 70)
    print()

    print(
        "[INFO] Starting fresh labeling pass."
    )

    print()

    # ========================================================
    # CLEAR OLD CLASSIFICATIONS
    # ========================================================

    print(
        "[CLEANUP] Removing old classifications..."
    )

    clear_directory(
        label_dir
    )

    clear_directory(
        preview_dir
    )

    clear_directory(
        negative_review_dir
    )

    print()

    print(
        "[CLEANUP] Old classifications cleared."
    )

    print(
        "[CLEANUP] Organized images were NOT touched."
    )

    print()

    # ========================================================
    # CONFIGURATION
    # ========================================================

    print(
        f"[MODEL]             {model_path}"
    )

    print(
        f"[IMAGES]            {image_dir}"
    )

    print(
        f"[LABELS]            {label_dir}"
    )

    print(
        f"[REVIEW]            {preview_dir}"
    )

    print(
        f"[NO DETECTION]      {negative_review_dir}"
    )

    print(
        f"[AUTO CONFIDENCE]   {args.confidence:.2f}"
    )

    print(
        f"[REVIEW CONFIDENCE] {args.review_confidence:.2f}"
    )

    print()

    # ========================================================
    # FIND IMAGES
    # ========================================================

    images = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in IMAGE_EXTENSIONS
    )

    if not images:

        print(
            "[ERROR] No images found."
        )

        return

    print(
        f"[INFO] Found {len(images)} images."
    )

    print()

    # ========================================================
    # COUNT DATASET TYPES
    # ========================================================

    expected_negative_count = sum(
        1
        for image in images
        if is_negative_image(image)
    )

    expected_positive_count = (
        len(images)
        - expected_negative_count
    )

    print(
        f"[DATASET] Positive images: "
        f"{expected_positive_count}"
    )

    print(
        f"[DATASET] Negative images: "
        f"{expected_negative_count}"
    )

    print()

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "[MODEL] Loading YOLO model..."
    )

    model = YOLO(
        str(model_path)
    )

    print(
        "[MODEL] Model loaded."
    )

    print()

    # ========================================================
    # COUNTERS
    # ========================================================

    total = 0

    positive_images = 0
    negative_images = 0

    negative_labels_created = 0

    auto_labeled = 0
    needs_review = 0
    no_detection = 0

    errors = 0

    # ========================================================
    # PROCESS
    # ========================================================

    for image_path in images:

        total += 1

        print(
            f"[{total}/{len(images)}] "
            f"{image_path.name}"
        )

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        is_negative = (
            is_negative_image(
                image_path
            )
        )

        # ====================================================
        # NEGATIVE IMAGE
        # ====================================================

        if is_negative:

            negative_images += 1

            print(
                "    [TYPE] NEGATIVE"
            )

            print(
                "    [YOLO] SKIPPED"
            )

            create_empty_label(
                label_path
            )

            negative_labels_created += 1

            print(
                "    [LABEL] Empty label created."
            )

            print(
                "    [OK] Negative registered."
            )

            continue

        # ====================================================
        # POSITIVE IMAGE
        # ====================================================

        positive_images += 1

        print(
            "    [TYPE] POSITIVE"
        )

        try:

            # ------------------------------------------------
            # IMAGE SIZE
            # ------------------------------------------------

            with Image.open(
                image_path
            ) as image:

                img_w, img_h = (
                    image.size
                )

            # ------------------------------------------------
            # YOLO INFERENCE
            # ------------------------------------------------

            print(
                "    [YOLO] Running inference..."
            )

            results = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.review_confidence,
                verbose=False,
            )

            result = results[0]

            detections = []

            accepted_boxes = []

            # ------------------------------------------------
            # READ DETECTIONS
            # ------------------------------------------------

            if result.boxes is not None:

                for box in result.boxes:

                    confidence = float(
                        box.conf[0]
                    )

                    x1, y1, x2, y2 = (
                        box.xyxy[0].tolist()
                    )

                    detection = (
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence,
                    )

                    detections.append(
                        detection
                    )

                    if (
                        confidence
                        >= args.confidence
                    ):

                        accepted_boxes.append(
                            detection
                        )

            # =================================================
            # HIGH CONFIDENCE
            # =================================================

            if accepted_boxes:

                with open(
                    label_path,
                    "w",
                    encoding="utf-8",
                ) as file:

                    for detection in (
                        accepted_boxes
                    ):

                        (
                            x1,
                            y1,
                            x2,
                            y2,
                            confidence,
                        ) = detection

                        (
                            xc,
                            yc,
                            width,
                            height,
                        ) = yolo_box(
                            x1,
                            y1,
                            x2,
                            y2,
                            img_w,
                            img_h,
                        )

                        # Class 0 = weed_pen

                        file.write(
                            f"0 "
                            f"{xc:.6f} "
                            f"{yc:.6f} "
                            f"{width:.6f} "
                            f"{height:.6f}\n"
                        )

                auto_labeled += 1

                preview_path = (
                    preview_dir
                    / image_path.name
                )

                draw_boxes(
                    image_path,
                    detections,
                    preview_path,
                )

                print(
                    f"    [AUTO-LABEL] "
                    f"{len(accepted_boxes)} "
                    f"high-confidence box(es)."
                )

            # =================================================
            # LOW CONFIDENCE
            # =================================================

            elif detections:

                preview_path = (
                    preview_dir
                    / image_path.name
                )

                draw_boxes(
                    image_path,
                    detections,
                    preview_path,
                )

                needs_review += 1

                print(
                    "    [REVIEW] Detection found "
                    "below auto-label threshold."
                )

            # =================================================
            # NO DETECTION
            # =================================================

            else:

                no_detection += 1

                review_path = (
                    negative_review_dir
                    / image_path.name
                )

                copy_image(
                    image_path,
                    review_path,
                )

                print(
                    "    [NONE] No weed_pen detected."
                )

                print(
                    "    [REVIEW] Human review required."
                )

                print(
                    "    [NOTE] No label created."
                )

        except Exception as error:

            errors += 1

            print(
                f"    [ERROR] {error}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print(
        "                         SUMMARY"
    )
    print("=" * 70)
    print()

    print(
        f"Images found:              {total}"
    )

    print(
        f"Positive images:           {positive_images}"
    )

    print(
        f"Negative images:           {negative_images}"
    )

    print()

    print(
        f"Negative labels created:   "
        f"{negative_labels_created}"
    )

    print(
        f"Automatically labeled:     "
        f"{auto_labeled}"
    )

    print(
        f"Positive images to review: "
        f"{needs_review}"
    )

    print(
        f"Positive no detection:     "
        f"{no_detection}"
    )

    print(
        f"Errors:                    "
        f"{errors}"
    )

    print()

    print(
        "[OUTPUT]"
    )

    print(
        f"  Labels:          {label_dir}"
    )

    print(
        f"  Review:          {preview_dir}"
    )

    print(
        f"  No detection:    {negative_review_dir}"
    )

    print()

    print("=" * 70)
    print(
        "                         DONE"
    )
    print("=" * 70)
    print()

    print(
        "Dataset rules:"
    )

    print(
        "  negative_*  -> NO YOLO -> EMPTY .txt"
    )

    print(
        "  weed_pen_*  -> YOLO -> AUTO LABEL / REVIEW"
    )

    print()

    print(
        "Old classifications were cleared before processing."
    )

    print(
        "Organized images were preserved."
    )

    print()
    

if __name__ == "__main__":
    main()