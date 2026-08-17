"""
Dataset Statistics and Exploratory Data Analysis
Vehicle Insurance Fraud Detection

This script analyzes:
    - Class distribution
    - Train/Test distribution
    - Class percentages
    - Image dimensions
    - Minimum/maximum dimensions
    - Corrupted images
    - Duplicate images
    - Sample images
    - Dataset summary report
"""

from pathlib import Path
from collections import Counter, defaultdict
from hashlib import md5
import json

from PIL import Image, ImageOps, ImageDraw


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Car Insurance Fraud Detection"
    / "Insurance-Fraud-Detection"
    / "Insurance-Fraud-Detection"
)

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

SAMPLE_OUTPUT = PLOTS_DIR / "sample_images.jpg"
REPORT_OUTPUT = METRICS_DIR / "dataset_report.json"

CLASSES = ["Fraud", "Non-Fraud"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# DIRECTORY SETUP
# ============================================================

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_files(directory: Path):
    """
    Return all image files inside a directory.
    """

    if not directory.exists():
        return []

    return sorted(
        [
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def calculate_file_hash(path: Path):
    """
    Calculate MD5 hash of an image file.

    Used to detect exact duplicate files.
    """

    hash_md5 = md5()

    try:
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(8192), b""):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()

    except Exception:
        return None


def validate_image(path: Path):
    """
    Validate an image and return its dimensions.

    Returns:
        (True, width, height)
        or
        (False, None, None)
    """

    try:
        with Image.open(path) as image:
            image.verify()

        # Reopen because verify() invalidates the image object.
        with Image.open(path) as image:
            width, height = image.size

        return True, width, height

    except Exception:
        return False, None, None


def percentage(value, total):
    """
    Calculate percentage safely.
    """

    if total == 0:
        return 0.0

    return round((value / total) * 100, 2)


# ============================================================
# DATASET SCANNING
# ============================================================

def scan_dataset():
    """
    Scan the complete dataset and collect statistics.
    """

    print("\nScanning dataset...\n")

    records = []

    corrupt_images = []

    dimension_counter = Counter()

    hash_to_files = defaultdict(list)

    split_class_counts = {
        "train": {
            "Fraud": 0,
            "Non-Fraud": 0,
        },
        "test": {
            "Fraud": 0,
            "Non-Fraud": 0,
        },
    }

    split_valid_counts = {
        "train": 0,
        "test": 0,
    }

    # --------------------------------------------------------
    # TRAIN / TEST
    # --------------------------------------------------------

    for split_name, split_dir in [
        ("train", TRAIN_DIR),
        ("test", TEST_DIR),
    ]:

        print(f"Checking {split_name.upper()}...")

        for class_name in CLASSES:

            class_dir = split_dir / class_name

            image_files = get_image_files(class_dir)

            print(
                f"  {class_name:<12}: "
                f"{len(image_files)} files"
            )

            for image_path in image_files:

                is_valid, width, height = validate_image(image_path)

                relative_path = image_path.relative_to(PROJECT_ROOT)

                record = {
                    "path": str(relative_path),
                    "split": split_name,
                    "class": class_name,
                    "valid": is_valid,
                    "width": width,
                    "height": height,
                }

                records.append(record)

                if not is_valid:

                    corrupt_images.append(
                        str(relative_path)
                    )

                    continue

                split_class_counts[
                    split_name
                ][class_name] += 1

                split_valid_counts[
                    split_name
                ] += 1

                dimension_counter[
                    (width, height)
                ] += 1

                file_hash = calculate_file_hash(image_path)

                if file_hash:
                    hash_to_files[file_hash].append(
                        str(relative_path)
                    )

    return (
        records,
        corrupt_images,
        dimension_counter,
        hash_to_files,
        split_class_counts,
        split_valid_counts,
    )


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

def analyze_duplicates(hash_to_files):
    """
    Find exact duplicate images using file hashes.
    """

    duplicate_groups = []

    duplicate_image_count = 0

    for file_hash, files in hash_to_files.items():

        if len(files) > 1:

            duplicate_groups.append(
                {
                    "hash": file_hash,
                    "count": len(files),
                    "files": files,
                }
            )

            duplicate_image_count += len(files)

    return duplicate_groups, duplicate_image_count


# ============================================================
# DIMENSION ANALYSIS
# ============================================================

def analyze_dimensions(dimension_counter):
    """
    Calculate image dimension statistics.
    """

    if not dimension_counter:
        return {
            "unique_dimensions": 0,
            "most_common_dimensions": [],
            "min_width": None,
            "max_width": None,
            "min_height": None,
            "max_height": None,
        }

    all_dimensions = list(dimension_counter.keys())

    widths = [
        width
        for width, height in all_dimensions
    ]

    heights = [
        height
        for width, height in all_dimensions
    ]

    most_common = []

    for (width, height), count in dimension_counter.most_common(10):

        most_common.append(
            {
                "width": width,
                "height": height,
                "count": count,
            }
        )

    return {
        "unique_dimensions": len(dimension_counter),
        "most_common_dimensions": most_common,
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights),
    }


# ============================================================
# SAMPLE IMAGE GENERATION
# ============================================================

def create_sample_images(records, num_samples=12):
    """
    Create a simple contact sheet containing sample images.
    """

    valid_records = [
        record
        for record in records
        if record["valid"]
    ]

    if not valid_records:
        print("\nNo valid images available for sample generation.")

        return

    # Pick evenly distributed samples.
    step = max(
        len(valid_records) // num_samples,
        1,
    )

    selected_records = valid_records[::step][:num_samples]

    thumbnail_size = (220, 180)

    columns = 3

    rows = (
        len(selected_records) + columns - 1
    ) // columns

    canvas_width = columns * thumbnail_size[0]

    canvas_height = rows * (
        thumbnail_size[1] + 40
    )

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    for index, record in enumerate(selected_records):

        image_path = (
            PROJECT_ROOT
            / record["path"]
        )

        try:

            with Image.open(image_path) as image:

                image = image.convert("RGB")

                image.thumbnail(
                    thumbnail_size
                )

                x = (
                    index % columns
                ) * thumbnail_size[0]

                y = (
                    index // columns
                ) * (
                    thumbnail_size[1] + 40
                )

                paste_x = (
                    x
                    + (
                        thumbnail_size[0]
                        - image.width
                    )
                    // 2
                )

                paste_y = (
                    y
                    + (
                        thumbnail_size[1]
                        - image.height
                    )
                    // 2
                )

                canvas.paste(
                    image,
                    (paste_x, paste_y),
                )

                label = (
                    f"{record['split']} | "
                    f"{record['class']}"
                )

                draw.text(
                    (x + 5, y + thumbnail_size[1] + 5),
                    label,
                    fill="black",
                )

        except Exception:
            continue

    canvas.save(SAMPLE_OUTPUT)

    print(
        f"\nSample image grid saved to:"
        f"\n{SAMPLE_OUTPUT}"
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    split_class_counts,
    split_valid_counts,
    corrupt_images,
    duplicate_groups,
    duplicate_image_count,
    dimension_stats,
):
    """
    Print human-readable dataset summary.
    """

    train_fraud = split_class_counts["train"]["Fraud"]
    train_nonfraud = split_class_counts["train"]["Non-Fraud"]

    test_fraud = split_class_counts["test"]["Fraud"]
    test_nonfraud = split_class_counts["test"]["Non-Fraud"]

    train_total = (
        train_fraud
        + train_nonfraud
    )

    test_total = (
        test_fraud
        + test_nonfraud
    )

    total = (
        train_total
        + test_total
    )

    print("\n")
    print("=" * 45)
    print("             DATASET SUMMARY")
    print("=" * 45)

    print("\nTRAIN")

    print(
        f"Fraud       : {train_fraud}"
    )

    print(
        f"Non-Fraud   : {train_nonfraud}"
    )

    print(
        f"Total       : {train_total}"
    )

    print("\nTRAIN CLASS PERCENTAGE")

    print(
        f"Fraud       : "
        f"{percentage(train_fraud, train_total)}%"
    )

    print(
        f"Non-Fraud   : "
        f"{percentage(train_nonfraud, train_total)}%"
    )

    print("\nTEST")

    print(
        f"Fraud       : {test_fraud}"
    )

    print(
        f"Non-Fraud   : {test_nonfraud}"
    )

    print(
        f"Total       : {test_total}"
    )

    print("\nTEST CLASS PERCENTAGE")

    print(
        f"Fraud       : "
        f"{percentage(test_fraud, test_total)}%"
    )

    print(
        f"Non-Fraud   : "
        f"{percentage(test_nonfraud, test_total)}%"
    )

    print("\nTOTAL")

    print(
        f"Total Images: {total}"
    )

    print("\nOVERALL CLASS DISTRIBUTION")

    overall_fraud = (
        train_fraud
        + test_fraud
    )

    overall_nonfraud = (
        train_nonfraud
        + test_nonfraud
    )

    print(
        f"Fraud       : "
        f"{overall_fraud} "
        f"({percentage(overall_fraud, total)}%)"
    )

    print(
        f"Non-Fraud   : "
        f"{overall_nonfraud} "
        f"({percentage(overall_nonfraud, total)}%)"
    )

    print("\nIMAGE VALIDATION")

    print(
        f"Corrupted images : "
        f"{len(corrupt_images)}"
    )

    print("\nDUPLICATE ANALYSIS")

    print(
        f"Duplicate groups : "
        f"{len(duplicate_groups)}"
    )

    print(
        f"Images in duplicate groups : "
        f"{duplicate_image_count}"
    )

    print("\nIMAGE DIMENSIONS")

    print(
        f"Unique dimensions : "
        f"{dimension_stats['unique_dimensions']}"
    )

    print(
        f"Minimum width    : "
        f"{dimension_stats['min_width']}"
    )

    print(
        f"Maximum width    : "
        f"{dimension_stats['max_width']}"
    )

    print(
        f"Minimum height   : "
        f"{dimension_stats['min_height']}"
    )

    print(
        f"Maximum height   : "
        f"{dimension_stats['max_height']}"
    )

    print("\nMOST COMMON DIMENSIONS")

    for dimension in dimension_stats[
        "most_common_dimensions"
    ]:

        print(
            f"{dimension['width']} x "
            f"{dimension['height']} "
            f"-> {dimension['count']} images"
        )

    print("\n" + "=" * 45)


# ============================================================
# SAVE JSON REPORT
# ============================================================

def save_report(
    split_class_counts,
    split_valid_counts,
    corrupt_images,
    duplicate_groups,
    duplicate_image_count,
    dimension_stats,
):
    """
    Save all important statistics as JSON.
    """

    train_fraud = split_class_counts["train"]["Fraud"]
    train_nonfraud = split_class_counts["train"]["Non-Fraud"]

    test_fraud = split_class_counts["test"]["Fraud"]
    test_nonfraud = split_class_counts["test"]["Non-Fraud"]

    train_total = (
        train_fraud
        + train_nonfraud
    )

    test_total = (
        test_fraud
        + test_nonfraud
    )

    total = train_total + test_total

    overall_fraud = (
        train_fraud
        + test_fraud
    )

    overall_nonfraud = (
        train_nonfraud
        + test_nonfraud
    )

    report = {

        "dataset": {
            "name": "Insurance-Fraud-Detection",
            "total_images": total,
        },

        "train": {
            "Fraud": train_fraud,
            "Non-Fraud": train_nonfraud,
            "total": train_total,
            "Fraud_percentage": percentage(
                train_fraud,
                train_total,
            ),
            "Non-Fraud_percentage": percentage(
                train_nonfraud,
                train_total,
            ),
        },

        "test": {
            "Fraud": test_fraud,
            "Non-Fraud": test_nonfraud,
            "total": test_total,
            "Fraud_percentage": percentage(
                test_fraud,
                test_total,
            ),
            "Non-Fraud_percentage": percentage(
                test_nonfraud,
                test_total,
            ),
        },

        "overall_class_distribution": {
            "Fraud": overall_fraud,
            "Non-Fraud": overall_nonfraud,
            "Fraud_percentage": percentage(
                overall_fraud,
                total,
            ),
            "Non-Fraud_percentage": percentage(
                overall_nonfraud,
                total,
            ),
        },

        "validation": {
            "corrupted_images": len(
                corrupt_images
            ),
            "corrupted_files": corrupt_images,
        },

        "duplicates": {
            "duplicate_groups": len(
                duplicate_groups
            ),
            "images_in_duplicate_groups":
                duplicate_image_count,
            "groups": duplicate_groups,
        },

        "dimensions": dimension_stats,
    }

    with open(
        REPORT_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        f"\nDataset report saved to:"
        f"\n{REPORT_OUTPUT}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 45)
    print(" VEHICLE INSURANCE FRAUD DETECTION - EDA")
    print("=" * 45)

    print(
        f"\nDataset path:\n{DATASET_DIR}"
    )

    if not DATASET_DIR.exists():

        print(
            "\nERROR: Dataset directory does not exist!"
        )

        print(
            "\nExpected:"
        )

        print(
            DATASET_DIR
        )

        return

    (
        records,
        corrupt_images,
        dimension_counter,
        hash_to_files,
        split_class_counts,
        split_valid_counts,
    ) = scan_dataset()

    duplicate_groups, duplicate_image_count = (
        analyze_duplicates(
            hash_to_files
        )
    )

    dimension_stats = analyze_dimensions(
        dimension_counter
    )

    print_summary(
        split_class_counts,
        split_valid_counts,
        corrupt_images,
        duplicate_groups,
        duplicate_image_count,
        dimension_stats,
    )

    create_sample_images(
        records,
        num_samples=12,
    )

    save_report(
        split_class_counts,
        split_valid_counts,
        corrupt_images,
        duplicate_groups,
        duplicate_image_count,
        dimension_stats,
    )

    print(
        "\nEDA completed successfully! ✅"
    )


if __name__ == "__main__":
    main()