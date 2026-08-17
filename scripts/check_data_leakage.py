from pathlib import Path
import hashlib
from collections import Counter

import pandas as pd
from PIL import Image
import numpy as np


# ============================================================
# CONFIG
# ============================================================

TRAIN_CSV = Path("data/splits/train.csv")
VAL_CSV = Path("data/splits/val.csv")
TEST_CSV = Path("data/splits/test.csv")

# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
}


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    """Calculate SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    try:
        with open(path, "rb") as f:

            while True:

                chunk = f.read(1024 * 1024)

                if not chunk:
                    break

                sha256.update(chunk)

        return sha256.hexdigest()

    except Exception as e:

        print(
            f"WARNING: Could not hash {path}: {e}"
        )

        return None


def normalize_path(path):
    """Normalize path for comparison."""

    return str(
        Path(path).resolve()
    ).lower()


def get_filename(path):
    return Path(path).name.lower()


def get_stem(path):
    return Path(path).stem.lower()


# ============================================================
# LOAD SPLITS
# ============================================================

def load_split(csv_path, split_name):

    if not csv_path.exists():

        raise FileNotFoundError(
            f"{split_name} CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = {
        "path",
        "label"
    }

    missing = required_columns - set(
        df.columns
    )

    if missing:

        raise ValueError(
            f"{split_name} is missing columns: "
            f"{missing}"
        )

    df["path"] = df["path"].astype(str)

    df["label"] = df["label"].astype(int)

    return df


# ============================================================
# PATH OVERLAP
# ============================================================

def check_path_overlap(
    datasets
):

    print()
    print("=" * 60)
    print("1. EXACT PATH OVERLAP")
    print("=" * 60)

    path_sets = {}

    for name, df in datasets.items():

        path_sets[name] = {
            normalize_path(p)
            for p in df["path"]
        }

    pairs = [
        ("Train", "Validation"),
        ("Train", "Test"),
        ("Validation", "Test")
    ]

    leakage_found = False

    for a, b in pairs:

        overlap = (
            path_sets[a]
            &
            path_sets[b]
        )

        print(
            f"{a} ↔ {b}: {len(overlap)}"
        )

        if overlap:

            leakage_found = True

            print(
                "  Examples:"
            )

            for item in list(overlap)[:5]:

                print(
                    f"    {item}"
                )

    return leakage_found


# ============================================================
# FILENAME OVERLAP
# ============================================================

def check_filename_overlap(
    datasets
):

    print()
    print("=" * 60)
    print("2. FILENAME OVERLAP")
    print("=" * 60)

    filename_sets = {}

    for name, df in datasets.items():

        filename_sets[name] = {
            get_filename(p)
            for p in df["path"]
        }

    pairs = [
        ("Train", "Validation"),
        ("Train", "Test"),
        ("Validation", "Test")
    ]

    suspicious = False

    for a, b in pairs:

        overlap = (
            filename_sets[a]
            &
            filename_sets[b]
        )

        print(
            f"{a} ↔ {b}: {len(overlap)}"
        )

        if overlap:

            suspicious = True

            print(
                "  Examples:"
            )

            for item in list(overlap)[:10]:

                print(
                    f"    {item}"
                )

    return suspicious


# ============================================================
# SHA-256 OVERLAP
# ============================================================

def build_hash_map(df):

    result = {}

    for path in df["path"]:

        p = Path(path)

        if not p.exists():

            print(
                f"WARNING: File does not exist:"
                f" {path}"
            )

            continue

        file_hash = sha256_file(p)

        if file_hash is not None:

            result.setdefault(
                file_hash,
                []
            ).append(
                str(p)
            )

    return result


def check_hash_overlap(
    datasets
):

    print()
    print("=" * 60)
    print("3. SHA-256 IMAGE DUPLICATE CHECK")
    print("=" * 60)

    hashes = {}

    for name, df in datasets.items():

        print(
            f"Hashing {name} images..."
        )

        hashes[name] = build_hash_map(
            df
        )

        print(
            f"  Unique hashes: "
            f"{len(hashes[name])}"
        )

    pairs = [
        ("Train", "Validation"),
        ("Train", "Test"),
        ("Validation", "Test")
    ]

    leakage_found = False

    for a, b in pairs:

        overlap = (
            set(hashes[a])
            &
            set(hashes[b])
        )

        print()
        print(
            f"{a} ↔ {b}: "
            f"{len(overlap)} identical images"
        )

        if overlap:

            leakage_found = True

            print(
                "  LEAKAGE FOUND!"
            )

            for h in list(overlap)[:5]:

                print(
                    f"\n  Hash: {h}"
                )

                print(
                    f"    {a}: "
                    f"{hashes[a][h][:2]}"
                )

                print(
                    f"    {b}: "
                    f"{hashes[b][h][:2]}"
                )

    return leakage_found


# ============================================================
# DUPLICATES INSIDE EACH SPLIT
# ============================================================

def check_internal_duplicates(
    datasets
):

    print()
    print("=" * 60)
    print("4. DUPLICATES INSIDE EACH SPLIT")
    print("=" * 60)

    suspicious = False

    for name, df in datasets.items():

        duplicate_paths = (
            df["path"]
            .astype(str)
            .str.lower()
            .duplicated()
            .sum()
        )

        duplicate_filenames = (
            df["path"]
            .map(get_filename)
            .duplicated()
            .sum()
        )

        print()
        print(
            f"{name}:"
        )

        print(
            f"  Duplicate paths     : "
            f"{duplicate_paths}"
        )

        print(
            f"  Duplicate filenames : "
            f"{duplicate_filenames}"
        )

        if (
            duplicate_paths > 0
            or
            duplicate_filenames > 0
        ):

            suspicious = True

    return suspicious


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

def check_label_distribution(
    datasets
):

    print()
    print("=" * 60)
    print("5. LABEL DISTRIBUTION")
    print("=" * 60)

    for name, df in datasets.items():

        counts = (
            df["label"]
            .value_counts()
            .sort_index()
        )

        total = len(df)

        print()
        print(
            f"{name}:"
        )

        for label, count in counts.items():

            percentage = (
                count /
                total *
                100
            )

            label_name = (
                "Non-Fraud"
                if label == 0
                else "Fraud"
            )

            print(
                f"  {label_name:<10}: "
                f"{count:>5} "
                f"({percentage:.2f}%)"
            )


# ============================================================
# CROSS-SPLIT DIRECTORY CHECK
# ============================================================

def check_directory_overlap(
    datasets
):

    print()
    print("=" * 60)
    print("6. DIRECTORY OVERLAP")
    print("=" * 60)

    directories = {}

    for name, df in datasets.items():

        directories[name] = Counter(
            str(
                Path(p).parent.resolve()
            ).lower()
            for p in df["path"]
        )

    all_dirs = set()

    for values in directories.values():

        all_dirs.update(
            values.keys()
        )

    suspicious = False

    for directory in sorted(all_dirs):

        present_in = [
            name
            for name in datasets
            if directory
            in directories[name]
        ]

        if len(present_in) > 1:

            suspicious = True

            print()
            print(
                "Directory appears in:"
            )

            print(
                f"  {directory}"
            )

            print(
                f"  Splits: "
                f"{', '.join(present_in)}"
            )

    if not suspicious:

        print(
            "No directory-level overlap detected."
        )

    return suspicious


# ============================================================
# IMAGE METADATA CHECK
# ============================================================

def inspect_image_sizes(
    datasets
):

    print()
    print("=" * 60)
    print("7. IMAGE METADATA SANITY CHECK")
    print("=" * 60)

    for name, df in datasets.items():

        widths = []
        heights = []
        modes = []

        for path in df["path"].head(100):

            try:

                with Image.open(path) as image:

                    widths.append(
                        image.width
                    )

                    heights.append(
                        image.height
                    )

                    modes.append(
                        image.mode
                    )

            except Exception:

                continue

        if not widths:

            print(
                f"{name}: Could not inspect images."
            )

            continue

        print()
        print(
            f"{name} "
            f"(first {len(widths)} images):"
        )

        print(
            f"  Width range  : "
            f"{min(widths)} - {max(widths)}"
        )

        print(
            f"  Height range : "
            f"{min(heights)} - {max(heights)}"
        )

        print(
            f"  Modes        : "
            f"{set(modes)}"
        )


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 60)
    print("VEHICLE FRAUD DATA LEAKAGE AUDIT")
    print("=" * 60)

    print()

    print(
        "This script does NOT modify any data."
    )

    print(
        "It only checks the existing splits."
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_df = load_split(
        TRAIN_CSV,
        "Train"
    )

    val_df = load_split(
        VAL_CSV,
        "Validation"
    )

    test_df = load_split(
        TEST_CSV,
        "Test"
    )

    datasets = {
        "Train": train_df,
        "Validation": val_df,
        "Test": test_df
    }

    # --------------------------------------------------------
    # Basic dataset information
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    for name, df in datasets.items():

        print(
            f"{name:<12}: "
            f"{len(df)} samples"
        )

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    path_leakage = check_path_overlap(
        datasets
    )

    filename_suspicious = (
        check_filename_overlap(
            datasets
        )
    )

    hash_leakage = check_hash_overlap(
        datasets
    )

    internal_duplicates = (
        check_internal_duplicates(
            datasets
        )
    )

    check_label_distribution(
        datasets
    )

    directory_suspicious = (
        check_directory_overlap(
            datasets
        )
    )

    inspect_image_sizes(
        datasets
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL LEAKAGE AUDIT RESULT")
    print("=" * 60)

    print()

    if not path_leakage:

        print(
            "✓ No exact path overlap"
        )

    else:

        print(
            "❌ EXACT PATH LEAKAGE DETECTED"
        )

    if not hash_leakage:

        print(
            "✓ No SHA-256 duplicate leakage"
        )

    else:

        print(
            "❌ IDENTICAL IMAGE LEAKAGE DETECTED"
        )

    if not filename_suspicious:

        print(
            "✓ No filename overlap"
        )

    else:

        print(
            "⚠ Filename overlap detected"
        )

    if not internal_duplicates:

        print(
            "✓ No obvious internal duplicates"
        )

    else:

        print(
            "⚠ Internal duplicates detected"
        )

    if not directory_suspicious:

        print(
            "✓ No directory overlap"
        )

    else:

        print(
            "⚠ Directory overlap detected"
        )

    print()

    if (
        path_leakage
        or
        hash_leakage
    ):

        print(
            "❌ DATA LEAKAGE DETECTED"
        )

        print(
            "Do NOT trust the current test metrics."
        )

    elif (
        filename_suspicious
        or
        internal_duplicates
        or
        directory_suspicious
    ):

        print(
            "⚠ NO DIRECT LEAKAGE FOUND,"
        )

        print(
            "BUT SUSPICIOUS STRUCTURE REQUIRES "
            "FURTHER INVESTIGATION."
        )

    else:

        print(
            "✓ NO OBVIOUS DATA LEAKAGE DETECTED"
        )

        print(
            "The split passes the basic leakage audit."
        )

    print()
    print("=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    main()