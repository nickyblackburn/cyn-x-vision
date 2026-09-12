from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def yolo_box(x1, y1, x2, y2, img_w, img_h):
    """Convert pixel coordinates to normalized YOLO format."""

    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h

    return x_center, y_center, width, height


def draw_boxes(image_path, detections, output_path):
    """Create a visual preview of the predicted bounding boxes."""

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for detection in detections:
        x1, y1, x2, y2, confidence = detection

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=4,
        )

        label = f"weed_pen {confidence:.2f}"

        draw.rectangle(
            [x1, max(0, y1 - 25), x1 + 150, y1],
            fill="red",
        )

        draw.text(
            (x1 + 5, max(0, y1 - 22)),
            label,
            fill="white",
        )

    image.save(output_path, quality=95)


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate YOLO bounding-box labels."
    )

    parser.add_argument(
        "--model",
        default="models/best.pt",
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
        help="Directory where YOLO labels will be written.",
    )

    parser.add_argument(
        "--preview",
        default="dataset/review",
        help="Directory for images with predicted boxes.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.70,
        help="Minimum confidence required for an automatic label.",
    )

    parser.add_argument(
        "--review-confidence",
        type=float,
        default=0.50,
        help="Minimum confidence for a detection to appear in review.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size.",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    preview_dir = Path(args.preview)

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    if not image_dir.exists():
        print(f"[ERROR] Image directory not found: {image_dir}")
        return

    label_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("          CYN-X VISION AUTO LABELER")
    print("=" * 60)
    print()

    print(f"[MODEL]      {model_path}")
    print(f"[IMAGES]     {image_dir}")
    print(f"[LABELS]     {label_dir}")
    print(f"[PREVIEW]    {preview_dir}")
    print(f"[CONFIDENCE] {args.confidence}")
    print()

    model = YOLO(str(model_path))

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"[INFO] Found {len(images)} images.")
    print()

    total = 0
    labeled = 0
    reviewed = 0
    no_detection = 0

    for image_path in images:

        total += 1

        print(
            f"[{total}/{len(images)}] "
            f"{image_path.name}"
        )

        try:
            with Image.open(image_path) as image:
                img_w, img_h = image.size

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

                    confidence = float(box.conf[0])

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

                    if confidence >= args.confidence:

                        accepted_boxes.append(
                            (
                                x1,
                                y1,
                                x2,
                                y2,
                                confidence,
                            )
                        )

            label_path = label_dir / (
                image_path.stem + ".txt"
            )

            # -------------------------------------------------
            # HIGH CONFIDENCE
            # Automatically create YOLO labels.
            # -------------------------------------------------

            if accepted_boxes:

                with open(label_path, "w", encoding="utf-8") as f:

                    for x1, y1, x2, y2, confidence in accepted_boxes:

                        xc, yc, w, h = yolo_box(
                            x1,
                            y1,
                            x2,
                            y2,
                            img_w,
                            img_h,
                        )

                        # Single class:
                        # 0 = weed_pen

                        f.write(
                            f"0 {xc:.6f} {yc:.6f} "
                            f"{w:.6f} {h:.6f}\n"
                        )

                labeled += 1

                # Save preview
                preview_path = preview_dir / image_path.name

                draw_boxes(
                    image_path,
                    detections,
                    preview_path,
                )

                print(
                    f"    [OK] {len(accepted_boxes)} "
                    f"high-confidence box(es)"
                )

            # -------------------------------------------------
            # LOW/MEDIUM CONFIDENCE
            # Save for human review.
            # -------------------------------------------------

            elif detections:

                preview_path = preview_dir / image_path.name

                draw_boxes(
                    image_path,
                    detections,
                    preview_path,
                )

                reviewed += 1

                print(
                    "    [REVIEW] Detection found, "
                    "but confidence is below threshold."
                )

            else:

                no_detection += 1

                print(
                    "    [NONE] No detection."
                )

        except Exception as e:

            print(
                f"    [ERROR] {e}"
            )

    print()
    print("=" * 60)
    print("                 SUMMARY")
    print("=" * 60)

    print(f"Images processed:       {total}")
    print(f"Automatically labeled:  {labeled}")
    print(f"Needs review:           {reviewed}")
    print(f"No detection:           {no_detection}")

    print()
    print(f"Labels:   {label_dir}")
    print(f"Reviews:  {preview_dir}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()