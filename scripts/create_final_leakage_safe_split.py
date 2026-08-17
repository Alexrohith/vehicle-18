from pathlib import Path
from collections import defaultdict
import json
import random

import pandas as pd
import imagehash
from PIL import Image, ImageFile
from sklearn.model_selection import GroupShuffleSplit

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# CONFIG
# ============================================================

OLD_TRAIN = Path("data/splits/train.csv")
OLD_VAL = Path("data/splits/val.csv")
OLD_TEST = Path("data/splits/test.csv")

OUTPUT_DIR = Path("data/final_splits")

RANDOM_SEED = 42

# IMPORTANT:
# Keep this fixed once we validate it.
PHASH_THRESHOLD = 4

# Final proportions:
# 70% development
# 15% validation
# 15% final test
DEV_RATIO = 0.85
TEST_RATIO = 0.15

# Inside development:
# 70/85 = ~82.35% train
# 15/85 = ~17.65% validation


# ============================================================
# UNION FIND
# ============================================================

class UnionFind:

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):

        while self.parent[x] != x:

            self.parent[x] = self.parent[
                self.parent[x]
            ]

            x = self.parent[x]

        return x

    def union(self, a, b):

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb

        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra

        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ============================================================
# LOAD ALL IMAGES
# ============================================================

def load_all_images():

    train = pd.read_csv(OLD_TRAIN)
    val = pd.read_csv(OLD_VAL)
    test = pd.read_csv(OLD_TEST)

    train["old_split"] = "train"
    val["old_split"] = "validation"
    test["old_split"] = "test"

    df = pd.concat(
        [train, val, test],
        ignore_index=True
    )

    df = df[
        ["path", "label", "old_split"]
    ].drop_duplicates(
        subset=["path"]
    ).reset_index(drop=True)

    return df


# ============================================================
# HASH
# ============================================================

def calculate_hashes(df):

    print("=" * 60)
    print("CALCULATING pHASH VALUES")
    print("=" * 60)

    hashes = []

    for i, path in enumerate(df["path"]):

        try:

            with Image.open(path) as image:

                image = image.convert("RGB")

                h = imagehash.phash(image)

            hashes.append(h)

        except Exception as e:

            raise RuntimeError(
                f"Cannot process image:\n{path}\n{e}"
            )

        if (
            (i + 1) % 250 == 0
            or
            i + 1 == len(df)
        ):

            print(
                f"Processed "
                f"{i + 1}/{len(df)}"
            )

    return hashes


# ============================================================
# BUILD GROUPS
# ============================================================

def build_groups(df, hashes):

    print()
    print("=" * 60)
    print("BUILDING DUPLICATE GROUPS")
    print("=" * 60)

    n = len(df)

    uf = UnionFind(n)

    matches = 0

    # --------------------------------------------------------
    # Pairwise pHash comparison
    # --------------------------------------------------------

    for i in range(n):

        for j in range(i + 1, n):

            distance = (
                hashes[i] -
                hashes[j]
            )

            if distance <= PHASH_THRESHOLD:

                uf.union(i, j)

                matches += 1

        if (
            (i + 1) % 250 == 0
            or
            i + 1 == n
        ):

            print(
                f"Compared "
                f"{i + 1}/{n} | "
                f"Links: {matches}"
            )

    # --------------------------------------------------------
    # Convert roots to group IDs
    # --------------------------------------------------------

    root_to_group = {}

    group_ids = []

    next_group_id = 0

    for i in range(n):

        root = uf.find(i)

        if root not in root_to_group:

            root_to_group[root] = next_group_id

            next_group_id += 1

        group_ids.append(
            root_to_group[root]
        )

    df = df.copy()

    df["group_id"] = group_ids

    print()
    print(
        f"Total images : {n}"
    )

    print(
        f"Near-duplicate links : {matches}"
    )

    print(
        f"Unique groups : "
        f"{df['group_id'].nunique()}"
    )

    return df


# ============================================================
# GROUP STATISTICS
# ============================================================

def print_cross_split_groups(df):

    print()
    print("=" * 60)
    print("CHECKING ORIGINAL SPLIT CONTAMINATION")
    print("=" * 60)

    cross_split = 0

    for group_id, group in df.groupby(
        "group_id"
    ):

        splits = set(
            group["old_split"]
        )

        if len(splits) > 1:

            cross_split += 1

    print(
        f"Groups crossing old splits: "
        f"{cross_split}"
    )

    return cross_split


# ============================================================
# GROUP-AWARE FINAL SPLIT
# ============================================================

def create_final_split(df):

    print()
    print("=" * 60)
    print("CREATING FINAL GROUP-AWARE SPLIT")
    print("=" * 60)

    # ========================================================
    # STEP 1
    #
    # 85% DEVELOPMENT
    # 15% FINAL TEST
    #
    # GROUPS CANNOT BE SPLIT.
    # ========================================================

    splitter_test = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED
    )

    dev_idx, test_idx = next(
        splitter_test.split(
            df,
            groups=df["group_id"]
        )
    )

    development = df.iloc[
        dev_idx
    ].copy()

    final_test = df.iloc[
        test_idx
    ].copy()

    # ========================================================
    # STEP 2
    #
    # DEVELOPMENT:
    # 82.35% TRAIN
    # 17.65% VALIDATION
    #
    # This gives approximately:
    #
    # TRAIN      70%
    # VALIDATION 15%
    # TEST       15%
    # ========================================================

    validation_ratio_inside_dev = (
        0.15 / 0.85
    )

    splitter_val = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_ratio_inside_dev,
        random_state=RANDOM_SEED
    )

    train_idx, val_idx = next(
        splitter_val.split(
            development,
            groups=development["group_id"]
        )
    )

    train = development.iloc[
        train_idx
    ].copy()

    validation = development.iloc[
        val_idx
    ].copy()

    train["split"] = "train"
    validation["split"] = "validation"
    final_test["split"] = "test"

    return (
        train,
        validation,
        final_test
    )


# ============================================================
# VERIFY GROUP ISOLATION
# ============================================================

def verify_group_isolation(
    train,
    validation,
    test
):

    print()
    print("=" * 60)
    print("VERIFYING GROUP ISOLATION")
    print("=" * 60)

    train_groups = set(
        train["group_id"]
    )

    val_groups = set(
        validation["group_id"]
    )

    test_groups = set(
        test["group_id"]
    )

    checks = {

        "Train ↔ Validation":
            train_groups & val_groups,

        "Train ↔ Test":
            train_groups & test_groups,

        "Validation ↔ Test":
            val_groups & test_groups
    }

    for name, overlap in checks.items():

        print(
            f"{name}: {len(overlap)}"
        )

        if overlap:

            raise RuntimeError(
                f"LEAKAGE DETECTED: {name}"
            )

    print()
    print(
        "✓ ZERO GROUP OVERLAP"
    )


# ============================================================
# VERIFY PATHS
# ============================================================

def verify_paths(
    train,
    validation,
    test
):

    print()
    print("=" * 60)
    print("VERIFYING EXACT PATH ISOLATION")
    print("=" * 60)

    train_paths = set(
        train["path"]
    )

    val_paths = set(
        validation["path"]
    )

    test_paths = set(
        test["path"]
    )

    overlaps = {

        "Train ↔ Validation":
            train_paths & val_paths,

        "Train ↔ Test":
            train_paths & test_paths,

        "Validation ↔ Test":
            val_paths & test_paths
    }

    for name, overlap in overlaps.items():

        print(
            f"{name}: {len(overlap)}"
        )

        if overlap:

            raise RuntimeError(
                f"PATH LEAKAGE: {name}"
            )

    print()
    print(
        "✓ ZERO PATH OVERLAP"
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def print_distribution(
    train,
    validation,
    test
):

    print()
    print("=" * 60)
    print("FINAL DATA DISTRIBUTION")
    print("=" * 60)

    datasets = {

        "TRAIN": train,

        "VALIDATION": validation,

        "FINAL TEST": test
    }

    for name, data in datasets.items():

        fraud = int(
            (data["label"] == 1).sum()
        )

        non_fraud = int(
            (data["label"] == 0).sum()
        )

        total = len(data)

        print()
        print(name)

        print(
            f"Total      : {total}"
        )

        print(
            f"Non-Fraud  : "
            f"{non_fraud} "
            f"({non_fraud / total * 100:.2f}%)"
        )

        print(
            f"Fraud      : "
            f"{fraud} "
            f"({fraud / total * 100:.2f}%)"
        )

        print(
            f"Groups     : "
            f"{data['group_id'].nunique()}"
        )


# ============================================================
# SAVE
# ============================================================

def save_splits(
    train,
    validation,
    test
):

    print()
    print("=" * 60)
    print("SAVING FINAL SPLITS")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train[
        ["path", "label"]
    ].to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    validation[
        ["path", "label"]
    ].to_csv(
        OUTPUT_DIR / "val.csv",
        index=False
    )

    test[
        ["path", "label"]
    ].to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    print(
        "✓ train.csv"
    )

    print(
        "✓ val.csv"
    )

    print(
        "✓ test.csv"
    )


# ============================================================
# SAVE GROUP MANIFEST
# ============================================================

def save_manifest(df):

    manifest_path = (
        OUTPUT_DIR /
        "group_manifest.csv"
    )

    df[
        [
            "path",
            "label",
            "group_id",
            "old_split"
        ]
    ].to_csv(
        manifest_path,
        index=False
    )

    print()
    print(
        f"Group manifest saved:"
    )

    print(
        manifest_path
    )


# ============================================================
# SAVE AUDIT
# ============================================================

def save_audit(
    df,
    train,
    validation,
    test,
    cross_split_groups
):

    audit = {

        "dataset_total": len(df),

        "random_seed": RANDOM_SEED,

        "phash_threshold": PHASH_THRESHOLD,

        "total_groups":
            int(df["group_id"].nunique()),

        "original_cross_split_groups":
            int(cross_split_groups),

        "train_samples": len(train),

        "validation_samples":
            len(validation),

        "test_samples":
            len(test),

        "train_fraud":
            int(train["label"].sum()),

        "validation_fraud":
            int(validation["label"].sum()),

        "test_fraud":
            int(test["label"].sum()),

        "test_locked": True,

        "test_used_for_model_selection": False,

        "test_used_for_threshold_selection": False
    }

    path = (
        OUTPUT_DIR /
        "final_split_audit.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            audit,
            f,
            indent=4
        )

    print()
    print(
        f"Audit saved:"
    )

    print(path)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("FINAL LEAKAGE-SAFE DATASET CREATION")
    print("=" * 60)

    print()
    print(
        "THIS SCRIPT DOES NOT MODIFY:"
    )

    print(
        "data/splits/"
    )

    print()
    print(
        "A NEW FINAL DATASET WILL BE CREATED:"
    )

    print(
        "data/final_splits/"
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_all_images()

    print()
    print(
        f"Loaded {len(df)} unique images."
    )

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    missing = [
        p for p in df["path"]
        if not Path(p).exists()
    ]

    if missing:

        raise FileNotFoundError(
            f"{len(missing)} image paths are missing."
        )

    print(
        "✓ All image paths exist."
    )

    # --------------------------------------------------------
    # Hash
    # --------------------------------------------------------

    hashes = calculate_hashes(
        df
    )

    # --------------------------------------------------------
    # Groups
    # --------------------------------------------------------

    df = build_groups(
        df,
        hashes
    )

    # --------------------------------------------------------
    # Original contamination
    # --------------------------------------------------------

    cross_split_groups = (
        print_cross_split_groups(df)
    )

    # --------------------------------------------------------
    # Final split
    # --------------------------------------------------------

    (
        train,
        validation,
        test
    ) = create_final_split(df)

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_group_isolation(
        train,
        validation,
        test
    )

    verify_paths(
        train,
        validation,
        test
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_distribution(
        train,
        validation,
        test
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_splits(
        train,
        validation,
        test
    )

    save_manifest(
        df
    )

    save_audit(
        df,
        train,
        validation,
        test,
        cross_split_groups
    )

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL SPLIT CREATION COMPLETE")
    print("=" * 60)

    print()
    print(
        "TRAIN + VALIDATION = DEVELOPMENT DATA"
    )

    print(
        "TEST = PERMANENTLY HELD-OUT DATA"
    )

    print()
    print(
        "Do NOT train the final model yet."
    )

    print(
        "First verify the generated statistics."
    )


if __name__ == "__main__":
    main()