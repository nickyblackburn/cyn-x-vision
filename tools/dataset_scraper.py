#!/usr/bin/env python3
"""
CYN-X Vision - Weed Pen Dataset Scraper

Collects publicly indexed images for a YOLO object-detection dataset.

Install:
    python -m pip install -U ddgs requests pillow

Example:
    python dataset_scraper.py --count 500 --out ..\\dataset

Test first:
    python dataset_scraper.py --count 25 --out ..\\dataset
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError
from ddgs import DDGS


# ------------------------------------------------------------
# Search queries
# ------------------------------------------------------------

DEFAULT_QUERIES = [
    "weed pen",
    "THC vape pen",
    "cannabis vape pen",
    "weed vape cartridge",
    "THC cartridge",
    "cannabis cartridge",
    "510 vape cartridge",
    "cannabis disposable vape",
    "THC disposable vape",
    "cannabis vape device",
]


# ------------------------------------------------------------
# HTTP configuration
# ------------------------------------------------------------

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/153.0 Safari/537.36"
)


# ------------------------------------------------------------
# Hashing
# ------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    """Return SHA-256 hash for an image's raw bytes."""
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------
# File extension
# ------------------------------------------------------------

def safe_extension(content_type: str, url: str) -> str:
    """
    Determine a reasonable image extension from the HTTP content type
    or the URL.
    """

    content_type = content_type.lower().split(";")[0].strip()

    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tif",
    }

    if content_type in mapping:
        return mapping[content_type]

    suffix = Path(urlparse(url).path).suffix.lower()

    allowed = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
    }

    if suffix in allowed:
        if suffix == ".jpeg":
            return ".jpg"

        if suffix == ".tiff":
            return ".tif"

        return suffix

    return ".jpg"


# ------------------------------------------------------------
# Image validation
# ------------------------------------------------------------

def validate_image(
    data: bytes,
    min_width: int,
    min_height: int,
) -> tuple[bool, int, int, str]:
    """
    Check whether downloaded bytes are a valid image and meet
    minimum dimensions.

    Returns:
        (valid, width, height, format)
    """

    try:
        # First verify the image isn't corrupted.
        with Image.open(io.BytesIO(data)) as image:
            image.verify()

        # Re-open because verify() invalidates the image object.
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            image_format = (image.format or "").lower()

        if width < min_width:
            return False, width, height, image_format

        if height < min_height:
            return False, width, height, image_format

        return True, width, height, image_format

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        return False, 0, 0, ""


# ------------------------------------------------------------
# Download
# ------------------------------------------------------------

def download_image(
    session: requests.Session,
    url: str,
    timeout: int,
) -> tuple[bytes | None, str]:
    """
    Download an image.

    Returns:
        (bytes, content_type)
    """

    try:
        response = session.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            },
            allow_redirects=True,
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "content-type",
            "",
        )

        # Don't save HTML pages pretending to be images.
        if not content_type.lower().startswith("image/"):
            return None, content_type

        return response.content, content_type

    except requests.RequestException:
        return None, ""


# ------------------------------------------------------------
# Existing dataset
# ------------------------------------------------------------

def load_existing_metadata(
    metadata_path: Path,
) -> tuple[set[str], set[str], int]:
    """
    Load hashes and URLs from a previous run.

    This allows the scraper to resume instead of downloading
    everything again.
    """

    seen_hashes: set[str] = set()
    seen_urls: set[str] = set()
    count = 0

    if not metadata_path.exists():
        return seen_hashes, seen_urls, count

    try:
        with metadata_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                digest = row.get("sha256", "").strip()
                image_url = row.get("image_url", "").strip()

                if digest:
                    seen_hashes.add(digest)

                if image_url:
                    seen_urls.add(image_url)

                count += 1

    except OSError as exc:
        print(f"[WARN] Could not read metadata: {exc}")

    return seen_hashes, seen_urls, count


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description="CYN-X Vision image dataset scraper."
    )

    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Target number of unique images.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset"),
        help="Dataset output directory.",
    )

    parser.add_argument(
        "--per-query",
        type=int,
        default=100,
        help="Maximum search results per query.",
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
        "--timeout",
        type=int,
        default=15,
        help="HTTP download timeout in seconds.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Delay between downloads.",
    )

    parser.add_argument(
        "--queries",
        nargs="*",
        default=None,
        help="Override the default search queries.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    out_dir = args.out

    images_dir = out_dir / "images"

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = out_dir / "metadata.csv"

    # --------------------------------------------------------
    # Queries
    # --------------------------------------------------------

    queries = args.queries or DEFAULT_QUERIES

    # --------------------------------------------------------
    # HTTP session
    # --------------------------------------------------------

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
        }
    )

    # --------------------------------------------------------
    # Resume existing dataset
    # --------------------------------------------------------

    seen_hashes, seen_urls, metadata_count = (
        load_existing_metadata(metadata_path)
    )

    existing_files = list(images_dir.glob("*"))

    saved = len(existing_files)

    # If metadata contains more records than the directory scan,
    # use the larger number for informational purposes.
    if metadata_count > saved:
        saved = metadata_count

    # --------------------------------------------------------
    # Metadata CSV
    # --------------------------------------------------------

    write_header = not metadata_path.exists()

    metadata_file = metadata_path.open(
        "a",
        encoding="utf-8",
        newline="",
    )

    writer = csv.DictWriter(
        metadata_file,
        fieldnames=[
            "filename",
            "sha256",
            "width",
            "height",
            "format",
            "query",
            "title",
            "source",
            "image_url",
            "page_url",
        ],
    )

    if write_header:
        writer.writeheader()

    # --------------------------------------------------------
    # Startup information
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(" CYN-X VISION DATASET SCRAPER")
    print("=" * 60)
    print(f"Target images : {args.count}")
    print(f"Existing      : {saved}")
    print(f"Output        : {images_dir}")
    print(f"Metadata      : {metadata_path}")
    print(f"Queries       : {len(queries)}")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    try:

        with DDGS() as ddgs:

            for query in queries:

                if saved >= args.count:
                    break

                print(f"[SEARCH] {query}")

                try:

                    results = list(
                        ddgs.images(
                            query=query,
                            region="us-en",
                            safesearch="moderate",
                            max_results=args.per_query,
                            backend="auto",
                        )
                    )

                except Exception as exc:

                    print(
                        f"[WARN] Search failed for "
                        f"'{query}': {exc}"
                    )

                    continue

                # Mix results so we don't always take the same
                # ordering from the search engine.
                random.shuffle(results)

                print(
                    f"[INFO] Found {len(results)} candidates."
                )

                # ------------------------------------------------
                # Process search results
                # ------------------------------------------------

                for result in results:

                    if saved >= args.count:
                        break

                    image_url = result.get("image")

                    if not image_url:
                        continue

                    # URL duplicate.
                    if image_url in seen_urls:
                        continue

                    seen_urls.add(image_url)

                    # ------------------------------------------------
                    # Download
                    # ------------------------------------------------

                    data, content_type = download_image(
                        session=session,
                        url=image_url,
                        timeout=args.timeout,
                    )

                    if not data:
                        continue

                    # ------------------------------------------------
                    # Hash duplicate
                    # ------------------------------------------------

                    digest = sha256_bytes(data)

                    if digest in seen_hashes:
                        continue

                    # ------------------------------------------------
                    # Validate image
                    # ------------------------------------------------

                    valid, width, height, image_format = (
                        validate_image(
                            data=data,
                            min_width=args.min_width,
                            min_height=args.min_height,
                        )
                    )

                    if not valid:
                        continue

                    # ------------------------------------------------
                    # Filename
                    # ------------------------------------------------

                    extension = safe_extension(
                        content_type=content_type,
                        url=image_url,
                    )

                    filename = (
                        f"weed_pen_{saved + 1:05d}"
                        f"{extension}"
                    )

                    filepath = images_dir / filename

                    # ------------------------------------------------
                    # Save
                    # ------------------------------------------------

                    try:

                        filepath.write_bytes(data)

                    except OSError as exc:

                        print(
                            f"[WARN] Could not save "
                            f"{filepath}: {exc}"
                        )

                        continue

                    # ------------------------------------------------
                    # Register image
                    # ------------------------------------------------

                    seen_hashes.add(digest)

                    saved += 1

                    writer.writerow(
                        {
                            "filename": filename,
                            "sha256": digest,
                            "width": width,
                            "height": height,
                            "format": image_format,
                            "query": query,
                            "title": result.get(
                                "title",
                                "",
                            ),
                            "source": result.get(
                                "source",
                                "",
                            ),
                            "image_url": image_url,
                            "page_url": result.get(
                                "url",
                                "",
                            ),
                        }
                    )

                    metadata_file.flush()

                    print(
                        f"[{saved:04d}/{args.count}] "
                        f"{filename} "
                        f"{width}x{height}"
                    )

                    # ------------------------------------------------
                    # Be polite to image hosts.
                    # ------------------------------------------------

                    time.sleep(args.delay)

    except KeyboardInterrupt:

        print()
        print("[STOP] Interrupted by user.")

    finally:

        metadata_file.close()

    # ------------------------------------------------------------
    # Finished
    # ------------------------------------------------------------

    print()
    print("=" * 60)
    print(" CYN-X VISION SCRAPER FINISHED")
    print("=" * 60)
    print(f"Images saved : {saved}")
    print(f"Image folder : {images_dir}")
    print(f"Metadata     : {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()