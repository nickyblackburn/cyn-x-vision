#!/usr/bin/env python3
"""
CYN-X Vision - Dataset Scraper

Collects publicly indexed images for a YOLO object-detection dataset.

The scraper supports two dataset categories:

    POSITIVE
        Images expected to contain a weed pen.

    NEGATIVE
        Ordinary images that should NOT contain a weed pen.

Negative images are intentionally saved without YOLO labels here.
The labeling/preparation pipeline will create/use empty label files
for reviewed negative images.

Install:
    python -m pip install -U ddgs requests pillow

Examples:

    # Collect 100 positive + 40 negative images
    python tools\\dataset_scraper.py --positive 100 --negative 40

    # Small test
    python tools\\dataset_scraper.py --positive 10 --negative 5

    # Custom output directory
    python tools\\dataset_scraper.py ^
        --positive 100 ^
        --negative 40 ^
        --out dataset
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


# ============================================================
# SEARCH QUERIES
# ============================================================

POSITIVE_QUERIES = [
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


# These are intentionally ordinary/background subjects.
#
# IMPORTANT:
# Search results still need human review.
# A search for "person" can contain someone holding a vape.
#

NEGATIVE_QUERIES = [
    "person portrait",
    "person sitting indoors",
    "person at desk",
    "person using computer",
    "person gaming",
    "hand holding phone",
    "computer desk",
    "gaming setup",
    "keyboard desk",
    "computer keyboard",
    "laptop desk",
    "living room",
    "bedroom",
    "office desk",
    "ordinary pen",
    "ballpoint pen",
    "pencil",
    "smartphone",
    "game controller",
    "headphones",
]


# ============================================================
# HTTP CONFIGURATION
# ============================================================

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/153.0 Safari/537.36"
)


# ============================================================
# HASHING
# ============================================================

def sha256_bytes(data: bytes) -> str:
    """Return SHA-256 hash for an image's raw bytes."""
    return hashlib.sha256(data).hexdigest()


# ============================================================
# FILE EXTENSION
# ============================================================

def safe_extension(content_type: str, url: str) -> str:
    """
    Determine a reasonable image extension from HTTP content type
    or the URL.
    """

    content_type = (
        content_type.lower()
        .split(";")[0]
        .strip()
    )

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

    suffix = Path(
        urlparse(url).path
    ).suffix.lower()

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


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(
    data: bytes,
    min_width: int,
    min_height: int,
) -> tuple[bool, int, int, str]:

    """
    Verify that downloaded bytes are a valid image and meet
    minimum dimensions.

    Returns:

        (valid, width, height, format)
    """

    try:

        # Verify that the image isn't corrupted.
        with Image.open(io.BytesIO(data)) as image:
            image.verify()

        # verify() invalidates the image, so reopen it.
        with Image.open(io.BytesIO(data)) as image:

            width, height = image.size

            image_format = (
                image.format or ""
            ).lower()

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


# ============================================================
# DOWNLOAD
# ============================================================

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
                "Accept": (
                    "image/avif,image/webp,"
                    "image/apng,image/svg+xml,"
                    "image/*,*/*;q=0.8"
                ),
            },
            allow_redirects=True,
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "content-type",
            "",
        )

        # Don't save HTML pretending to be an image.
        if not content_type.lower().startswith("image/"):
            return None, content_type

        return response.content, content_type

    except requests.RequestException:

        return None, ""


# ============================================================
# EXISTING DATASET
# ============================================================

def load_existing_metadata(
    metadata_path: Path,
) -> tuple[set[str], set[str], dict[str, int]]:

    """
    Load hashes, URLs, and category counters from previous runs.

    Older metadata files without a category column are treated
    as positive images for backwards compatibility.
    """

    seen_hashes: set[str] = set()
    seen_urls: set[str] = set()

    counters = {
        "positive": 0,
        "negative": 0,
    }

    if not metadata_path.exists():
        return seen_hashes, seen_urls, counters

    try:

        with metadata_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                digest = row.get(
                    "sha256",
                    "",
                ).strip()

                image_url = row.get(
                    "image_url",
                    "",
                ).strip()

                filename = row.get(
                    "filename",
                    "",
                ).strip()

                category = row.get(
                    "category",
                    "",
                ).strip().lower()

                if digest:
                    seen_hashes.add(digest)

                if image_url:
                    seen_urls.add(image_url)

                # ------------------------------------------------
                # Backwards compatibility
                # ------------------------------------------------

                if category not in {
                    "positive",
                    "negative",
                }:

                    if filename.startswith(
                        "negative_"
                    ):
                        category = "negative"

                    else:
                        category = "positive"

                counters[category] += 1

    except OSError as exc:

        print(
            f"[WARN] Could not read metadata: {exc}"
        )

    return seen_hashes, seen_urls, counters


# ============================================================
# NEXT FILENAME
# ============================================================

def make_filename(
    category: str,
    counters: dict[str, int],
    extension: str,
) -> str:

    """
    Generate a stable category-specific filename.
    """

    counters[category] += 1

    if category == "positive":

        prefix = "weed_pen"

    else:

        prefix = "negative"

    return (
        f"{prefix}_"
        f"{counters[category]:05d}"
        f"{extension}"
    )


# ============================================================
# SCRAPE CATEGORY
# ============================================================

def scrape_category(
    *,
    category: str,
    target: int,
    queries: list[str],
    args,
    session: requests.Session,
    seen_hashes: set[str],
    seen_urls: set[str],
    counters: dict[str, int],
    writer,
    metadata_file,
) -> int:

    """
    Scrape one category.

    Returns the number of newly saved images.
    """

    if target <= 0:
        return 0

    current = counters[category]

    if current >= target:

        print(
            f"[SKIP] {category}: "
            f"already have {current}/{target}"
        )

        return 0

    print()
    print("=" * 60)
    print(
        f" {category.upper()} DATASET"
    )
    print("=" * 60)
    print(
        f"Existing : {current}"
    )
    print(
        f"Target   : {target}"
    )
    print(
        f"Needed   : {target - current}"
    )
    print("=" * 60)
    print()

    saved_this_run = 0

    try:

        with DDGS() as ddgs:

            for query in queries:

                if counters[category] >= target:
                    break

                print(
                    f"[SEARCH/{category}] "
                    f"{query}"
                )

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

                random.shuffle(results)

                print(
                    f"[INFO] Found "
                    f"{len(results)} candidates."
                )

                for result in results:

                    if counters[category] >= target:
                        break

                    image_url = result.get(
                        "image"
                    )

                    if not image_url:
                        continue

                    # --------------------------------------------
                    # URL duplicate
                    # --------------------------------------------

                    if image_url in seen_urls:
                        continue

                    seen_urls.add(image_url)

                    # --------------------------------------------
                    # Download
                    # --------------------------------------------

                    data, content_type = (
                        download_image(
                            session=session,
                            url=image_url,
                            timeout=args.timeout,
                        )
                    )

                    if not data:
                        continue

                    # --------------------------------------------
                    # Hash duplicate
                    # --------------------------------------------

                    digest = sha256_bytes(data)

                    if digest in seen_hashes:
                        continue

                    # --------------------------------------------
                    # Validate
                    # --------------------------------------------

                    valid, width, height, image_format = (
                        validate_image(
                            data=data,
                            min_width=args.min_width,
                            min_height=args.min_height,
                        )
                    )

                    if not valid:
                        continue

                    # --------------------------------------------
                    # Filename
                    # --------------------------------------------

                    extension = safe_extension(
                        content_type=content_type,
                        url=image_url,
                    )

                    filename = make_filename(
                        category=category,
                        counters=counters,
                        extension=extension,
                    )

                    filepath = (
                        args.out
                        / "images"
                        / filename
                    )

                    # --------------------------------------------
                    # Save
                    # --------------------------------------------

                    try:

                        filepath.write_bytes(
                            data
                        )

                    except OSError as exc:

                        print(
                            f"[WARN] Could not save "
                            f"{filepath}: {exc}"
                        )

                        # Roll back counter because
                        # the image wasn't actually saved.
                        counters[category] -= 1

                        continue

                    # --------------------------------------------
                    # Register
                    # --------------------------------------------

                    seen_hashes.add(
                        digest
                    )

                    saved_this_run += 1

                    writer.writerow(
                        {
                            "filename": filename,
                            "category": category,
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
                        f"[{category.upper()} "
                        f"{counters[category]:04d}/"
                        f"{target}] "
                        f"{filename} "
                        f"{width}x{height}"
                    )

                    time.sleep(
                        args.delay
                    )

    except KeyboardInterrupt:

        raise

    return saved_this_run


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "CYN-X Vision positive/negative "
            "dataset scraper."
        )
    )

    parser.add_argument(
        "--positive",
        type=int,
        default=100,
        help=(
            "Target number of positive "
            "weed-pen images."
        ),
    )

    parser.add_argument(
        "--negative",
        type=int,
        default=40,
        help=(
            "Target number of negative "
            "ordinary/background images."
        ),
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
        help=(
            "Maximum search results "
            "per query."
        ),
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
        help=(
            "HTTP download timeout "
            "in seconds."
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help=(
            "Delay between downloads."
        ),
    )

    parser.add_argument(
        "--positive-queries",
        nargs="*",
        default=None,
        help=(
            "Override positive search queries."
        ),
    )

    parser.add_argument(
        "--negative-queries",
        nargs="*",
        default=None,
        help=(
            "Override negative search queries."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if args.positive < 0:
        parser.error(
            "--positive cannot be negative."
        )

    if args.negative < 0:
        parser.error(
            "--negative cannot be negative."
        )

    if (
        args.positive == 0
        and args.negative == 0
    ):
        parser.error(
            "At least one target must be greater than zero."
        )

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    images_dir = (
        args.out / "images"
    )

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        args.out / "metadata.csv"
    )

    # --------------------------------------------------------
    # Queries
    # --------------------------------------------------------

    positive_queries = (
        args.positive_queries
        if args.positive_queries is not None
        else POSITIVE_QUERIES
    )

    negative_queries = (
        args.negative_queries
        if args.negative_queries is not None
        else NEGATIVE_QUERIES
    )

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
    # Resume
    # --------------------------------------------------------

    seen_hashes, seen_urls, counters = (
        load_existing_metadata(
            metadata_path
        )
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    write_header = (
        not metadata_path.exists()
        or metadata_path.stat().st_size == 0
    )

    metadata_file = (
        metadata_path.open(
            "a",
            encoding="utf-8",
            newline="",
        )
    )

    writer = csv.DictWriter(
        metadata_file,
        fieldnames=[
            "filename",
            "category",
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
    # Startup
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        " CYN-X VISION DATASET SCRAPER"
    )
    print("=" * 60)

    print(
        f"Positive target : "
        f"{args.positive}"
    )

    print(
        f"Negative target : "
        f"{args.negative}"
    )

    print(
        f"Positive exist  : "
        f"{counters['positive']}"
    )

    print(
        f"Negative exist  : "
        f"{counters['negative']}"
    )

    print(
        f"Output          : "
        f"{images_dir}"
    )

    print(
        f"Metadata        : "
        f"{metadata_path}"
    )

    print("=" * 60)
    print()

    total_saved = 0

    try:

        # ====================================================
        # POSITIVE
        # ====================================================

        total_saved += scrape_category(
            category="positive",
            target=args.positive,
            queries=positive_queries,
            args=args,
            session=session,
            seen_hashes=seen_hashes,
            seen_urls=seen_urls,
            counters=counters,
            writer=writer,
            metadata_file=metadata_file,
        )

        # ====================================================
        # NEGATIVE
        # ====================================================

        total_saved += scrape_category(
            category="negative",
            target=args.negative,
            queries=negative_queries,
            args=args,
            session=session,
            seen_hashes=seen_hashes,
            seen_urls=seen_urls,
            counters=counters,
            writer=writer,
            metadata_file=metadata_file,
        )

    except KeyboardInterrupt:

        print()
        print(
            "[STOP] Interrupted by user."
        )

    finally:

        metadata_file.close()

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        " CYN-X VISION SCRAPER FINISHED"
    )
    print("=" * 60)

    print(
        f"Positive images : "
        f"{counters['positive']}"
    )

    print(
        f"Negative images : "
        f"{counters['negative']}"
    )

    print(
        f"New this run    : "
        f"{total_saved}"
    )

    print(
        f"Image folder    : "
        f"{images_dir}"
    )

    print(
        f"Metadata        : "
        f"{metadata_path}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()