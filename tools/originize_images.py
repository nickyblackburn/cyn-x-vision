#!/usr/bin/env python3
"""
CYN-X Vision - Image Organizer / Normalizer

Converts collected images into a consistent JPEG dataset suitable
for YOLO training.

Input:
    dataset/images/

Output:
    dataset/organized/
        weed_pen_00001.jpg
        weed_pen_00002.jpg
        ...

    dataset/rejected/
        corrupted_or_invalid_images

Features:
    - Converts PNG/WebP/BMP/TIFF/GIF/JPEG -> JPEG
    - Converts images to RGB
    - Handles images with transparency
    - Removes exact duplicates using SHA-256
    - Rejects corrupted images
    - Rejects images below minimum dimensions
    - Creates consistent filenames
    - Does NOT modify the original images
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}


def sha256_bytes(data: bytes) -> str:
    """Return SHA-256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def image_hash(image: Image.Image) -> str:
    """
    Create a normalized pixel hash.

    This catches duplicates even when the same image was downloaded
    in different formats.
    """

    image = image.convert("RGB")

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return sha256_bytes(buffer.getvalue())


def flatten_transparency(
    image: Image.Image,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Convert transparent images to RGB using a white background.
    """

    if image.mode in ("RGBA", "LA"):

        rgba = image.convert("RGBA")

        background_image = Image.new(
            "RGBA",
            rgba.size,
            background + (255,),
        )

        background_image.alpha_composite(rgba)

        return background_image.convert("RGB")

    if image.mode == "P":

        if "transparency" in image.info:

            rgba = image.convert("RGBA")

            background_image = Image.new(
                "RGBA",
                rgba.size,
                background + (255,),
            )

            background_image.alpha_composite(rgba)

            return background_image.convert("RGB")

    return image.convert("RGB")


def process_image(
    source: Path,
    destination: Path,
    min_width: int,
    min_height: int,
    jpeg_quality: int,
) -> tuple[bool, str, int, int]:
    """
    Convert one image to JPEG.

    Returns:

        success
        reason
        width
        height
    """

    try:

        with Image.open(source) as image:

            # Correct EXIF rotation.
            image = ImageOps.exif_transpose(image)

            width, height = image.size

            # Reject tiny images.
            if width < min_width:

                return (
                    False,
                    "too_small",
                    width,
                    height,
                )

            if height < min_height:

                return (
                    False,
                    "too_small",
                    width,
                    height,
                )

            # GIFs may contain multiple frames.
            # For dataset purposes, use the first frame.
            if getattr(image, "is_animated", False):

                image.seek(0)

            image = flatten_transparency(image)

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            image.save(
                destination,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,
                progressive=True,
            )

            return (
                True,
                "ok",
                width,
                height,
            )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:

        return (
            False,
            f"corrupted: {type(exc).__name__}",
            0,
            0,
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Normalize CYN-X Vision images into JPEG."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/images"),
        help="Directory containing scraped images.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/organized"),
        help="Directory for normalized JPEG images.",
    )

    parser.add_argument(
        "--rejected",
        type=Path,
        default=Path("dataset/rejected"),
        help="Directory for rejected images.",
    )

    parser.add_argument(
        "--min-width",
        type=int,
        default=400,
        help="Minimum image width.",
    )

    parser.add_argument(
        "--min-height",
        type=int,
        default=400,
        help="Minimum image height.",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality from 1-100.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Validate arguments
    # ---------------------------------------------------------

    if not args.input.exists():

        print(
            f"[ERROR] Input directory does not exist:"
            f"\n        {args.input}"
        )

        print(
            "\nRun the scraper first or provide --input."
        )

        return

    if not 1 <= args.quality <= 100:

        print(
            "[ERROR] --quality must be between 1 and 100."
        )

        return

    # ---------------------------------------------------------
    # Create directories
    # ---------------------------------------------------------

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.rejected.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Find images
    # ---------------------------------------------------------

    files = [
        path
        for path in args.input.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]

    files.sort()

    print()
    print("=" * 60)
    print(" CYN-X VISION IMAGE ORGANIZER")
    print("=" * 60)
    print(f"Input       : {args.input}")
    print(f"Output      : {args.output}")
    print(f"Rejected    : {args.rejected}")
    print(f"Images found: {len(files)}")
    print(f"Min size    : {args.min_width}x{args.min_height}")
    print(f"JPEG quality: {args.quality}")
    print("=" * 60)
    print()

    if not files:

        print(
            "[INFO] No supported images found."
        )

        return

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    converted = 0
    duplicates = 0
    rejected = 0
    errors = 0

    seen_hashes: set[str] = set()

    # ---------------------------------------------------------
    # Process
    # ---------------------------------------------------------

    for source in files:

        try:

            # Read image once to calculate normalized pixel hash.
            with Image.open(source) as image:

                image = ImageOps.exif_transpose(image)

                if getattr(image, "is_animated", False):

                    image.seek(0)

                normalized = flatten_transparency(image)

                width, height = normalized.size

                if width < args.min_width:
                    rejected += 1

                    destination = (
                        args.rejected
                        / source.name
                    )

                    shutil.copy2(
                        source,
                        destination,
                    )

                    print(
                        f"[REJECT] {source.name}"
                        f" - too small"
                    )

                    continue

                if height < args.min_height:
                    rejected += 1

                    destination = (
                        args.rejected
                        / source.name
                    )

                    shutil.copy2(
                        source,
                        destination,
                    )

                    print(
                        f"[REJECT] {source.name}"
                        f" - too small"
                    )

                    continue

                digest = image_hash(normalized)

                if digest in seen_hashes:

                    duplicates += 1

                    print(
                        f"[DUPLICATE] "
                        f"{source.name}"
                    )

                    continue

                seen_hashes.add(digest)

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ):

            rejected += 1
            errors += 1

            destination = (
                args.rejected
                / source.name
            )

            try:

                shutil.copy2(
                    source,
                    destination,
                )

            except OSError:
                pass

            print(
                f"[REJECT] {source.name}"
                f" - corrupted/unreadable"
            )

            continue

        # -----------------------------------------------------
        # Create sequential filename
        # -----------------------------------------------------

        filename = (
            f"weed_pen_{converted + 1:05d}.jpg"
        )

        destination = (
            args.output / filename
        )

        success, reason, width, height = process_image(
            source=source,
            destination=destination,
            min_width=args.min_width,
            min_height=args.min_height,
            jpeg_quality=args.quality,
        )

        if not success:

            rejected += 1

            print(
                f"[REJECT] {source.name}"
                f" - {reason}"
            )

            continue

        converted += 1

        print(
            f"[{converted:04d}] "
            f"{filename} "
            f"({width}x{height})"
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print(" ORGANIZATION COMPLETE")
    print("=" * 60)
    print(f"Converted    : {converted}")
    print(f"Duplicates   : {duplicates}")
    print(f"Rejected     : {rejected}")
    print(f"Errors       : {errors}")
    print(f"Output       : {args.output}")
    print(f"Rejected dir : {args.rejected}")
    print("=" * 60)


if __name__ == "__main__":
    main()