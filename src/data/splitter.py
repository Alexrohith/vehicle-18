from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    "data/raw/Car Insurance Fraud Detection/"
    "Insurance-Fraud-Detection/Insurance-Fraud-Detection"
)

OUTPUT_DIR = Path("data/splits")

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = {
    "Fraud": 1,
    "Non-Fraud": 0
}


# ============================================================
# COLLECT IMAGES
# ============================================================

def collect_images(directory: Path):
    records = []

    for class_name, label in CLASS_NAMES.items():

        class_dir = directory / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}"
            )

        for image_path in class_dir.rglob("*"):

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            records.append({
                "path": str(image_path),
                "label": label,
                "class_name": class_name
            })

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    train_dir = DATASET_ROOT / "train"
    test_dir = DATASET_ROOT / "test"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CREATING STRATIFIED DATA SPLITS")
    print("=" * 60)

    # --------------------------------------------------------
    # Collect original train images
    # --------------------------------------------------------

    train_records = collect_images(train_dir)

    df = pd.DataFrame(train_records)

    print(f"\nOriginal training images: {len(df)}")

    print("\nOriginal distribution:")
    print(df["class_name"].value_counts())

    # --------------------------------------------------------
    # Stratified train/validation split
    # --------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    # --------------------------------------------------------
    # Collect test images
    # --------------------------------------------------------

    test_records = collect_images(test_dir)

    test_df = pd.DataFrame(test_records)

    # --------------------------------------------------------
    # Reset indexes
    # --------------------------------------------------------

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # --------------------------------------------------------
    # Save CSV files
    # --------------------------------------------------------

    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    val_df.to_csv(
        OUTPUT_DIR / "val.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL SPLIT")
    print("=" * 60)

    print("\nTRAIN")
    print(f"Total: {len(train_df)}")
    print(train_df["class_name"].value_counts())

    print("\nVALIDATION")
    print(f"Total: {len(val_df)}")
    print(val_df["class_name"].value_counts())

    print("\nTEST")
    print(f"Total: {len(test_df)}")
    print(test_df["class_name"].value_counts())

    print("\n" + "=" * 60)
    print("SPLITS SAVED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()