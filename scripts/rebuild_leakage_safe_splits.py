from pathlib import Path
import shutil
import hashlib
import json
import random

import pandas as pd
from PIL import Image, ImageFile
import imagehash

from sklearn.model_selection import GroupShuffleSplit


ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# CONFIG
# ============================================================

TRAIN_CSV = Path("data/splits/train.csv")
VAL_CSV = Path("data/splits/val.csv")
TEST_CSV = Path("data/splits/test.csv")

OUTPUT_DIR = Path("data/splits_leakage_safe")

BACKUP_DIR = Path("data/splits_backup_before_leakage_fix")

RANDOM_SEED = 42

# pHash Hamming distance.
#
# 0 = extremely strong match
# 1-2 = very strong near-duplicate
# 3-4 = reasonably strong visual similarity
#
# We use 4 for the grouping stage.
PHASH_THRESHOLD = 4

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# UNION-FIND
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

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:

            self.parent[root_a] = root_b

        elif self.rank[root_a] > self.rank[root_b]:

            self.parent[root_b] = root_a

        else:

            self.parent[root_b] = root_a
            self.rank[root_a] += 1


# ============================================================
# LOAD EXISTING DATA
# ============================================================

def load_all_samples():

    print("=" * 60)
    print("LOADING EXISTING DATASET")
    print("=" * 60)

    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)

    train["original_split"] = "train"
    val["original_split"] = "validation"
    test["original_split"] = "test"

    df = pd.concat(
        [train, val, test],
        ignore_index=True
    )

    # Remove accidental duplicate rows
    df = df.drop_duplicates(
        subset=["path"]
    ).reset_index(drop=True)

    print()
    print(f"Original Train      : {len(train)}")
    print(f"Original Validation : {len(val)}")
    print(f"Original Test       : {len(test)}")
    print(f"Total unique images : {len(df)}")

    return df


# ============================================================
# BACKUP EXISTING SPLITS
# ============================================================

def backup_existing_splits():

    print()
    print("=" * 60)
    print("BACKING UP EXISTING SPLITS")
    print("=" * 60)

    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for csv_file in [
        TRAIN_CSV,
        VAL_CSV,
        TEST_CSV
    ]:

        destination = (
            BACKUP_DIR /
            csv_file.name
        )

        shutil.copy2(
            csv_file,
            destination
        )

        print(
            f"Backed up: {csv_file}"
        )


# ============================================================
# VALIDATE IMAGE PATHS
# ============================================================

def validate_paths(df):

    print()
    print("=" * 60)
    print("VALIDATING IMAGE PATHS")
    print("=" * 60)

    missing = []

    for i, path in enumerate(df["path"]):

        if not Path(path).exists():

            missing.append(path)

    if missing:

        print(
            f"\nERROR: {len(missing)} images are missing."
        )

        for path in missing[:20]:

            print(
                f"  {path}"
            )

        raise FileNotFoundError(
            "Missing image files detected."
        )

    print(
        f"✓ All {len(df)} image paths exist."
    )


# ============================================================
# CALCULATE PHASH
# ============================================================

def calculate_phash(path):

    try:

        with Image.open(path) as image:

            image = image.convert("RGB")

            return imagehash.phash(image)

    except Exception as e:

        print(
            f"\nWARNING: Could not hash {path}"
        )

        print(e)

        return None


# ============================================================
# HASH ALL IMAGES
# ============================================================

def calculate_all_hashes(df):

    print()
    print("=" * 60)
    print("CALCULATING PERCEPTUAL HASHES")
    print("=" * 60)

    hashes = []

    total = len(df)

    for i, path in enumerate(df["path"]):

        h = calculate_phash(path)

        if h is None:

            raise RuntimeError(
                f"Could not calculate pHash: {path}"
            )

        hashes.append(h)

        if (
            (i + 1) % 250 == 0
            or
            i + 1 == total
        ):

            print(
                f"Processed "
                f"{i + 1}/{total}"
            )

    return hashes


# ============================================================
# FIND NEAR-DUPLICATE GROUPS
# ============================================================

def build_duplicate_groups(df, hashes):

    print()
    print("=" * 60)
    print("BUILDING NEAR-DUPLICATE GROUPS")
    print("=" * 60)

    n = len(df)

    uf = UnionFind(n)

    comparisons = 0
    matches = 0

    print()
    print(
        f"Images to compare : {n}"
    )

    print(
        f"pHash threshold   : {PHASH_THRESHOLD}"
    )

    print()
    print(
        "Comparing image hashes..."
    )

    # --------------------------------------------------------
    # Pairwise comparison
    # --------------------------------------------------------

    for i in range(n):

        for j in range(i + 1, n):

            distance = (
                hashes[i] -
                hashes[j]
            )

            comparisons += 1

            if distance <= PHASH_THRESHOLD:

                uf.union(i, j)

                matches += 1

        if (
            (i + 1) % 250 == 0
            or
            i + 1 == n
        ):

            print(
                f"Compared row "
                f"{i + 1}/{n} | "
                f"Matches: {matches}"
            )

    # --------------------------------------------------------
    # Convert union-find roots → group IDs
    # --------------------------------------------------------

    root_to_group = {}

    group_ids = []

    next_group = 0

    for i in range(n):

        root = uf.find(i)

        if root not in root_to_group:

            root_to_group[root] = next_group

            next_group += 1

        group_ids.append(
            root_to_group[root]
        )

    df = df.copy()

    df["group_id"] = group_ids

    group_sizes = (
        df.groupby("group_id")
        .size()
    )

    duplicate_groups = (
        group_sizes[group_sizes > 1]
    )

    print()
    print(
        f"Total comparisons : {comparisons:,}"
    )

    print(
        f"Near-duplicate links : {matches:,}"
    )

    print(
        f"Total groups : {len(group_sizes):,}"
    )

    print(
        f"Groups containing >1 image : "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"Images belonging to duplicate groups : "
        f"{duplicate_groups.sum():,}"
    )

    return df


# ============================================================
# GROUP STATISTICS
# ============================================================

def print_group_statistics(df):

    print()
    print("=" * 60)
    print("DUPLICATE GROUP STATISTICS")
    print("=" * 60)

    group_info = (
        df.groupby("group_id")
        .agg(
            images=("path", "count"),
            fraud=("label", "sum"),
            original_splits=(
                "original_split",
                lambda x: ",".join(
                    sorted(set(x))
                )
            )
        )
        .reset_index()
    )

    suspicious = group_info[
        group_info["original_splits"].str.contains(
            ","
        )
    ]

    print()
    print(
        "Groups that currently cross "
        "Train/Validation/Test:"
    )

    print(
        len(suspicious)
    )

    if len(suspicious) > 0:

        print()

        print(
            suspicious.head(20).to_string(
                index=False
            )
        )

    return group_info


# ============================================================
# CREATE GROUP-AWARE SPLIT
# ============================================================

def create_group_split(df):

    print()
    print("=" * 60)
    print("CREATING GROUP-AWARE SPLIT")
    print("=" * 60)

    # --------------------------------------------------------
    # First split:
    #
    # 70% Train
    # 30% Temporary
    #
    # Groups cannot be divided.
    # --------------------------------------------------------

    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=(
            VAL_RATIO + TEST_RATIO
        ),
        random_state=RANDOM_SEED
    )

    train_idx, temp_idx = next(
        splitter_1.split(
            df,
            groups=df["group_id"]
        )
    )

    train_df = df.iloc[
        train_idx
    ].copy()

    temp_df = df.iloc[
        temp_idx
    ].copy()

    # --------------------------------------------------------
    # Second split:
    #
    # Half temporary → Validation
    # Half temporary → Test
    # --------------------------------------------------------

    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=RANDOM_SEED
    )

    val_idx, test_idx = next(
        splitter_2.split(
            temp_df,
            groups=temp_df["group_id"]
        )
    )

    val_df = temp_df.iloc[
        val_idx
    ].copy()

    test_df = temp_df.iloc[
        test_idx
    ].copy()

    # --------------------------------------------------------
    # Assign split names
    # --------------------------------------------------------

    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    return (
        train_df,
        val_df,
        test_df
    )


# ============================================================
# PRINT SPLIT STATISTICS
# ============================================================

def print_split_statistics(
    train_df,
    val_df,
    test_df
):

    print()
    print("=" * 60)
    print("NEW LEAKAGE-SAFE SPLIT")
    print("=" * 60)

    datasets = {
        "Train": train_df,
        "Validation": val_df,
        "Test": test_df
    }

    total = (
        len(train_df)
        +
        len(val_df)
        +
        len(test_df)
    )

    for name, data in datasets.items():

        fraud = int(
            (data["label"] == 1).sum()
        )

        non_fraud = int(
            (data["label"] == 0).sum()
        )

        print()
        print(name)

        print(
            f"  Total      : {len(data)}"
        )

        print(
            f"  Non-Fraud  : "
            f"{non_fraud} "
            f"({non_fraud / len(data) * 100:.2f}%)"
        )

        print(
            f"  Fraud      : "
            f"{fraud} "
            f"({fraud / len(data) * 100:.2f}%)"
        )

        print(
            f"  Groups     : "
            f"{data['group_id'].nunique()}"
        )

    print()
    print(
        f"Total images : {total}"
    )

    print()
    print(
        "Overall class distribution:"
    )

    all_fraud = int(
        (
            pd.concat(
                [
                    train_df,
                    val_df,
                    test_df
                ]
            )["label"]
            == 1
        ).sum()
    )

    print(
        f"  Fraud     : {all_fraud}"
    )

    print(
        f"  Non-Fraud : {total - all_fraud}"
    )


# ============================================================
# VERIFY GROUP ISOLATION
# ============================================================

def verify_group_isolation(
    train_df,
    val_df,
    test_df
):

    print()
    print("=" * 60)
    print("VERIFYING GROUP ISOLATION")
    print("=" * 60)

    train_groups = set(
        train_df["group_id"]
    )

    val_groups = set(
        val_df["group_id"]
    )

    test_groups = set(
        test_df["group_id"]
    )

    train_val = (
        train_groups &
        val_groups
    )

    train_test = (
        train_groups &
        test_groups
    )

    val_test = (
        val_groups &
        test_groups
    )

    print()
    print(
        f"Train ↔ Validation groups: "
        f"{len(train_val)}"
    )

    print(
        f"Train ↔ Test groups: "
        f"{len(train_test)}"
    )

    print(
        f"Validation ↔ Test groups: "
        f"{len(val_test)}"
    )

    if (
        train_val
        or train_test
        or val_test
    ):

        raise RuntimeError(
            "GROUP LEAKAGE DETECTED!"
        )

    print()
    print(
        "✓ No duplicate group appears "
        "in multiple splits."
    )


# ============================================================
# VERIFY PATH ISOLATION
# ============================================================

def verify_path_isolation(
    train_df,
    val_df,
    test_df
):

    print()
    print("=" * 60)
    print("VERIFYING PATH ISOLATION")
    print("=" * 60)

    train_paths = set(
        train_df["path"]
    )

    val_paths = set(
        val_df["path"]
    )

    test_paths = set(
        test_df["path"]
    )

    print(
        f"Train ↔ Validation: "
        f"{len(train_paths & val_paths)}"
    )

    print(
        f"Train ↔ Test: "
        f"{len(train_paths & test_paths)}"
    )

    print(
        f"Validation ↔ Test: "
        f"{len(val_paths & test_paths)}"
    )


# ============================================================
# SAVE SPLITS
# ============================================================

def save_splits(
    train_df,
    val_df,
    test_df
):

    print()
    print("=" * 60)
    print("SAVING LEAKAGE-SAFE SPLITS")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Only save the columns required
    # by the existing VehicleFraudDataset.

    train_output = train_df[
        ["path", "label"]
    ].copy()

    val_output = val_df[
        ["path", "label"]
    ].copy()

    test_output = test_df[
        ["path", "label"]
    ].copy()

    train_output.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    val_output.to_csv(
        OUTPUT_DIR / "val.csv",
        index=False
    )

    test_output.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    print(
        f"Saved: {OUTPUT_DIR / 'train.csv'}"
    )

    print(
        f"Saved: {OUTPUT_DIR / 'val.csv'}"
    )

    print(
        f"Saved: {OUTPUT_DIR / 'test.csv'}"
    )


# ============================================================
# SAVE AUDIT REPORT
# ============================================================

def save_audit_report(
    df,
    train_df,
    val_df,
    test_df
):

    report = {

        "random_seed": RANDOM_SEED,

        "phash_threshold": PHASH_THRESHOLD,

        "original_total_images": len(df),

        "new_split_sizes": {

            "train": len(train_df),

            "validation": len(val_df),

            "test": len(test_df)
        },

        "new_split_fraud_counts": {

            "train": int(
                train_df["label"].sum()
            ),

            "validation": int(
                val_df["label"].sum()
            ),

            "test": int(
                test_df["label"].sum()
            )
        },

        "total_groups": int(
            df["group_id"].nunique()
        ),

        "multi_image_groups": int(
            (
                df.groupby("group_id")
                .size()
                > 1
            ).sum()
        )
    }

    with open(
        OUTPUT_DIR / "leakage_safe_audit.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print()
    print(
        f"Audit report saved to:"
    )

    print(
        OUTPUT_DIR /
        "leakage_safe_audit.json"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LEAKAGE-SAFE DATASET REBUILD")
    print("=" * 60)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This script DOES NOT modify "
        "the existing data/splits files."
    )

    print(
        "A backup is created for safety."
    )

    print()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_all_samples()

    # --------------------------------------------------------
    # Backup
    # --------------------------------------------------------

    backup_existing_splits()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_paths(df)

    # --------------------------------------------------------
    # pHash
    # --------------------------------------------------------

    hashes = calculate_all_hashes(
        df
    )

    # --------------------------------------------------------
    # Groups
    # --------------------------------------------------------

    df = build_duplicate_groups(
        df,
        hashes
    )

    group_info = (
        print_group_statistics(df)
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_df,
        val_df,
        test_df
    ) = create_group_split(df)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_split_statistics(
        train_df,
        val_df,
        test_df
    )

    # --------------------------------------------------------
    # Verify group isolation
    # --------------------------------------------------------

    verify_group_isolation(
        train_df,
        val_df,
        test_df
    )

    # --------------------------------------------------------
    # Verify paths
    # --------------------------------------------------------

    verify_path_isolation(
        train_df,
        val_df,
        test_df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_splits(
        train_df,
        val_df,
        test_df
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    save_audit_report(
        df,
        train_df,
        val_df,
        test_df
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("LEAKAGE-SAFE REBUILD COMPLETE")
    print("=" * 60)

    print()
    print(
        "Existing splits were NOT overwritten."
    )

    print(
        "New splits are located at:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Backup of old splits:"
    )

    print(
        BACKUP_DIR
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Review the split statistics and "
        "leakage_safe_audit.json."
    )

    print(
        "Do NOT train the model yet."
    )


if __name__ == "__main__":

    main()