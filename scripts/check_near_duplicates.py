from pathlib import Path

import pandas as pd
from PIL import Image
import imagehash


# ============================================================
# CONFIG
# ============================================================

SPLITS = {
    "Train": "data/splits/train.csv",
    "Validation": "data/splits/val.csv",
    "Test": "data/splits/test.csv",
}

# Smaller = stricter
# 0 = effectively identical perceptual hash
# 1-5 = increasingly different
HAMMING_THRESHOLD = 6


# ============================================================
# LOAD DATA
# ============================================================

def load_split(csv_path):

    df = pd.read_csv(csv_path)

    return df["path"].astype(str).tolist()


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def calculate_hash(path):

    try:

        with Image.open(path) as image:

            image = image.convert("RGB")

            return imagehash.phash(
                image
            )

    except Exception as e:

        print(
            f"WARNING: Could not process {path}: {e}"
        )

        return None


# ============================================================
# BUILD HASHES
# ============================================================

def build_hashes(paths, name):

    hashes = {}

    print(
        f"\nHashing {name} images..."
    )

    for i, path in enumerate(paths):

        if i % 500 == 0:

            print(
                f"  Processed "
                f"{i}/{len(paths)}"
            )

        image_hash = calculate_hash(
            path
        )

        if image_hash is not None:

            hashes[path] = image_hash

    print(
        f"  Completed: "
        f"{len(hashes)} images"
    )

    return hashes


# ============================================================
# COMPARE SPLITS
# ============================================================

def compare_splits(
    hashes_a,
    hashes_b,
    name_a,
    name_b
):

    print()
    print("=" * 60)

    print(
        f"{name_a} ↔ {name_b}"
    )

    print("=" * 60)

    matches = []

    for path_a, hash_a in hashes_a.items():

        best_distance = 999

        best_path = None

        for path_b, hash_b in hashes_b.items():

            distance = (
                hash_a - hash_b
            )

            if distance < best_distance:

                best_distance = distance

                best_path = path_b

        if (
            best_distance
            <= HAMMING_THRESHOLD
        ):

            matches.append(
                (
                    path_a,
                    best_path,
                    best_distance
                )
            )

    print(
        f"Potential near-duplicates: "
        f"{len(matches)}"
    )

    if matches:

        print()

        for (
            path_a,
            path_b,
            distance
        ) in matches[:20]:

            print(
                f"Distance: {distance}"
            )

            print(
                f"  {name_a}:"
            )

            print(
                f"    {path_a}"
            )

            print(
                f"  {name_b}:"
            )

            print(
                f"    {path_b}"
            )

            print()

    return matches


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "VEHICLE FRAUD "
        "NEAR-DUPLICATE AUDIT"
    )

    print("=" * 60)

    print()

    print(
        f"Perceptual hash threshold: "
        f"{HAMMING_THRESHOLD}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    paths = {}

    for name, csv_path in SPLITS.items():

        paths[name] = load_split(
            csv_path
        )

        print(
            f"{name}: "
            f"{len(paths[name])} images"
        )

    # --------------------------------------------------------
    # Hash
    # --------------------------------------------------------

    hashes = {}

    for name in paths:

        hashes[name] = build_hashes(
            paths[name],
            name
        )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    train_val = compare_splits(
        hashes["Train"],
        hashes["Validation"],
        "Train",
        "Validation"
    )

    train_test = compare_splits(
        hashes["Train"],
        hashes["Test"],
        "Train",
        "Test"
    )

    val_test = compare_splits(
        hashes["Validation"],
        hashes["Test"],
        "Validation",
        "Test"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)

    print(
        "NEAR-DUPLICATE AUDIT SUMMARY"
    )

    print("=" * 60)

    print()

    print(
        f"Train ↔ Validation : "
        f"{len(train_val)}"
    )

    print(
        f"Train ↔ Test       : "
        f"{len(train_test)}"
    )

    print(
        f"Validation ↔ Test  : "
        f"{len(val_test)}"
    )

    print()

    total = (
        len(train_val)
        +
        len(train_test)
        +
        len(val_test)
    )

    if total == 0:

        print(
            "🟢 NO NEAR-DUPLICATES "
            "DETECTED."
        )

    else:

        print(
            "🟡 POTENTIAL NEAR-DUPLICATES "
            "DETECTED."
        )

        print()

        print(
            "These images require "
            "manual inspection."
        )

    print()
    print("=" * 60)


if __name__ == "__main__":

    main()