from __future__ import annotations

import argparse
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


def yolo_box(
    x1,
    y1,
    x2,
    y2,
    img_w,
    img_h,
):
    """Convert pixel coordinates to normalized YOLO format."""

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


def draw_boxes(
    image_path,
    detections,
    output_path,
):
    """Create a visual preview of predicted boxes."""

    image = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(image)

    for detection in detections:

        x1, y1, x2, y2, confidence = detection

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=4,
        )

        label = (
            f"weed_pen "
            f"{confidence:.2f}"
        )

        # Label background
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


def create_empty_label(label_path):
    """
    Create an empty YOLO label file.

    An empty label means:

        This image contains NO weed_pen.

    This should only be called after a human confirms
    that an image is genuinely negative.
    """

    label_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_path.write_text(
        "",
        encoding="utf-8",
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "CYN-X Vision automatic "
            "YOLO labeling tool."
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
        help="Directory containing images.",
    )

    parser.add_argument(
        "--labels",
        default="dataset/labels",
        help="Directory containing YOLO labels.",
    )

    parser.add_argument(
        "--preview",
        default="dataset/review",
        help="Directory for review images.",
    )

    parser.add_argument(
        "--negative-review",
        default="dataset/negative_review",
        help=(
            "Directory for images with "
            "no model detection."
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
            "Minimum confidence for "
            "a detection to be reviewed."
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow existing labels to be "
            "overwritten. DO NOT use this "
            "for human-created labels unless "
            "you intentionally want to replace them."
        ),
    )

    args = parser.parse_args()

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
    # CHECK PATHS
    # ========================================================

    if not model_path.exists():

        print(
            f"[ERROR] Model not found: "
            f"{model_path}"
        )

        return

    if not image_dir.exists():

        print(
            f"[ERROR] Image directory not found: "
            f"{image_dir}"
        )

        return

    label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    negative_review_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # STARTUP
    # ========================================================

    print("=" * 60)
    print(
        "          CYN-X VISION AUTO LABELER"
    )
    print("=" * 60)
    print()

    print(
        f"[MODEL]            {model_path}"
    )

    print(
        f"[IMAGES]           {image_dir}"
    )

    print(
        f"[LABELS]           {label_dir}"
    )

    print(
        f"[REVIEW]           {preview_dir}"
    )

    print(
        f"[NEGATIVE REVIEW]  {negative_review_dir}"
    )

    print(
        f"[AUTO CONFIDENCE]  {args.confidence}"
    )

    print(
        f"[REVIEW CONFIDENCE] {args.review_confidence}"
    )

    print()

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("[MODEL] Loading YOLO model...")

    model = YOLO(
        str(model_path)
    )

    print("[MODEL] Model loaded.")
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

    print(
        f"[INFO] Found {len(images)} images."
    )

    print()

    # ========================================================
    # COUNTERS
    # ========================================================

    total = 0
    skipped_existing = 0
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

        # ----------------------------------------------------
        # PROTECT EXISTING LABELS
        # ----------------------------------------------------

        if (
            label_path.exists()
            and not args.overwrite
        ):

            skipped_existing += 1

            print(
                "    [SKIP] Existing label "
                "protected."
            )

            continue

        try:

            # ------------------------------------------------
            # IMAGE SIZE
            # ------------------------------------------------

            with Image.open(
                image_path
            ) as image:

                img_w, img_h = image.size

            # ------------------------------------------------
            # YOLO INFERENCE
            # ------------------------------------------------

            results = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.review_confidence,
                verbose=False,
            )

            result = results[0]

            detections = []
            accepted_boxes = []

            if result.boxes is not None:

                for box in result.boxes:

                    confidence = float(
                        box.conf[0]
                    )

                    x1, y1, x2, y2 = (
                        box.xyxy[0].tolist()
                    )

                    detections.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                            confidence,
                        )
                    )

                    if (
                        confidence
                        >= args.confidence
                    ):

                        accepted_boxes.append(
                            (
                                x1,
                                y1,
                                x2,
                                y2,
                                confidence,
                            )
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

                    for (
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence,
                    ) in accepted_boxes:

                        (
                            xc,
                            yc,
                            w,
                            h,
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
                            f"{w:.6f} "
                            f"{h:.6f}\n"
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
                    f"    [OK] "
                    f"{len(accepted_boxes)} "
                    f"high-confidence "
                    f"box(es)"
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
                    "    [REVIEW] Detection "
                    "below auto-label "
                    "threshold."
                )

            # =================================================
            # NO DETECTION
            # =================================================

            else:

                no_detection += 1

                # --------------------------------------------
                # IMPORTANT:
                #
                # We DO NOT create an empty label here.
                #
                # A failed detection does NOT automatically
                # mean the image is a valid negative example.
                #
                # Human review must confirm it first.
                # --------------------------------------------

                negative_path = (
                    negative_review_dir
                    / image_path.name
                )

                # Copy image for human review.
                image_copy = Image.open(
                    image_path
                ).convert("RGB")

                image_copy.save(
                    negative_path,
                    quality=95,
                )

                print(
                    "    [NONE] No detection."
                )

                print(
                    "    [NEGATIVE REVIEW] "
                    "Human confirmation required."
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
    print("=" * 60)
    print(
        "                 SUMMARY"
    )
    print("=" * 60)

    print(
        f"Images processed:       {total}"
    )

    print(
        f"Existing labels skipped: {skipped_existing}"
    )

    print(
        f"Automatically labeled:   {auto_labeled}"
    )

    print(
        f"Needs review:            {needs_review}"
    )

    print(
        f"No detection:            {no_detection}"
    )

    print(
        f"Errors:                   {errors}"
    )

    print()

    print(
        f"Labels:           {label_dir}"
    )

    print(
        f"Detection review: {preview_dir}"
    )

    print(
        f"Negative review:   {negative_review_dir}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Images in negative_review need "
        "human confirmation before receiving "
        "an empty YOLO label."
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()