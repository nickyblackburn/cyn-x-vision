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


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def main():
    random.seed(SEED)

    if not SOURCE_IMAGES.exists():
        print(f"[ERROR] Image directory not found: {SOURCE_IMAGES}")
        return

    if not SOURCE_LABELS.exists():
        print(f"[ERROR] Label directory not found: {SOURCE_LABELS}")
        return

    images = sorted(
        p for p in SOURCE_IMAGES.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    # --------------------------------------------------------
    # Use every image that has a label file.
    #
    # IMPORTANT:
    # An EMPTY .txt file is intentional.
    # It means the image contains NO weed_pen.
    # --------------------------------------------------------

    dataset_images = []
    positive_images = []
    negative_images = []

    for image in images:

        label = SOURCE_LABELS / f"{image.stem}.txt"

        # No label file = not ready for training
        if not label.exists():
            continue

        dataset_images.append(image)

        # Empty label = negative/background image
        if label.stat().st_size == 0:
            negative_images.append(image)

        # Non-empty label = positive image
        else:
            positive_images.append(image)

    if len(dataset_images) < 2:
        print("[ERROR] Need at least 2 images with label files.")
        print()
        print("Every training image needs a matching .txt file.")
        print("An empty .txt file is valid for a negative image.")
        return

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.shuffle(dataset_images)

    split_index = max(
        1,
        int(len(dataset_images) * TRAIN_RATIO)
    )

    # Make sure validation isn't empty
    if split_index >= len(dataset_images):
        split_index = len(dataset_images) - 1

    train_images = dataset_images[:split_index]
    val_images = dataset_images[split_index:]

    # --------------------------------------------------------
    # Make directories
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Clear previous generated YOLO dataset
    #
    # This is SAFE because dataset/yolo is generated data.
    # Your original images and labels remain untouched.
    # --------------------------------------------------------

    for directory in [
        train_images_dir,
        val_images_dir,
        train_labels_dir,
        val_labels_dir,
    ]:
        for file in directory.iterdir():
            if file.is_file():
                file.unlink()

    # --------------------------------------------------------
    # Copy training images + labels
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Copy validation images + labels
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Count positives / negatives in each split
    # --------------------------------------------------------

    train_positive = sum(
        1 for image in train_images
        if (SOURCE_LABELS / f"{image.stem}.txt").stat().st_size > 0
    )

    train_negative = len(train_images) - train_positive

    val_positive = sum(
        1 for image in val_images
        if (SOURCE_LABELS / f"{image.stem}.txt").stat().st_size > 0
    )

    val_negative = len(val_images) - val_positive

    # --------------------------------------------------------
    # Create data.yaml
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("=" * 60)
    print("       CYN-X VISION DATASET PREPARED")
    print("=" * 60)
    print()

    print(f"Total images:      {len(dataset_images)}")
    print(f"Positive images:   {len(positive_images)}")
    print(f"Negative images:   {len(negative_images)}")
    print()

    print("TRAINING")
    print(f"  Images:          {len(train_images)}")
    print(f"  Positive:        {train_positive}")
    print(f"  Negative:        {train_negative}")
    print()

    print("VALIDATION")
    print(f"  Images:          {len(val_images)}")
    print(f"  Positive:        {val_positive}")
    print(f"  Negative:        {val_negative}")
    print()

    print(f"Dataset:           {DATASET}")
    print(f"Config:            {yaml_path}")
    print()

    print("Ready for YOLO training.")


if __name__ == "__main__":
    main()