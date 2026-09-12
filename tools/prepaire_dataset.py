from pathlib import Path
import random
import shutil

# ============================================================
# CYN-X VISION - PREPARE YOLO DATASET
# ============================================================

SOURCE_IMAGES = Path("dataset/organized")
SOURCE_LABELS = Path("dataset/labels")

DATASET = Path("dataset/yolo")

TRAIN_RATIO = 0.80
SEED = 42

CLASS_NAME = "weed_pen"


def main():
    random.seed(SEED)

    images = sorted(
        p for p in SOURCE_IMAGES.iterdir()
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".webp"
        }
    )

    # Only use images that actually have labels
    labeled_images = []

    for image in images:
        label = SOURCE_LABELS / f"{image.stem}.txt"

        if label.exists() and label.stat().st_size > 0:
            labeled_images.append(image)

    if len(labeled_images) < 2:
        print("[ERROR] Need at least 2 labeled images.")
        return

    random.shuffle(labeled_images)

    split_index = max(
        1,
        int(len(labeled_images) * TRAIN_RATIO)
    )

    train_images = labeled_images[:split_index]
    val_images = labeled_images[split_index:]

    # Make directories
    train_images_dir = DATASET / "images" / "train"
    val_images_dir = DATASET / "images" / "val"

    train_labels_dir = DATASET / "labels" / "train"
    val_labels_dir = DATASET / "labels" / "val"

    for directory in [
        train_images_dir,
        val_images_dir,
        train_labels_dir,
        val_labels_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    # Copy training images + labels
    for image in train_images:

        label = SOURCE_LABELS / f"{image.stem}.txt"

        shutil.copy2(
            image,
            train_images_dir / image.name
        )

        shutil.copy2(
            label,
            train_labels_dir / label.name
        )

    # Copy validation images + labels
    for image in val_images:

        label = SOURCE_LABELS / f"{image.stem}.txt"

        shutil.copy2(
            image,
            val_images_dir / image.name
        )

        shutil.copy2(
            label,
            val_labels_dir / label.name
        )

    # Create data.yaml
    yaml_path = DATASET / "data.yaml"

    yaml_path.write_text(
        f"""path: {DATASET.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: {CLASS_NAME}
""",
        encoding="utf-8"
    )

    print("=" * 60)
    print("       CYN-X VISION DATASET PREPARED")
    print("=" * 60)
    print()
    print(f"Total labeled: {len(labeled_images)}")
    print(f"Training:      {len(train_images)}")
    print(f"Validation:    {len(val_images)}")
    print()
    print(f"Dataset:       {DATASET}")
    print(f"Config:        {yaml_path}")
    print()
    print("Ready for YOLO training.")


if __name__ == "__main__":
    main()