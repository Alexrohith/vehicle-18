from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.preprocessing import (
    VehicleFraudDataset,
    train_transform,
    eval_transform,
)


# ============================================================
# CONFIG
# ============================================================

TRAIN_CSV = Path("data/final_splits/train.csv")
VAL_CSV = Path("data/final_splits/val.csv")
TEST_CSV = Path("data/final_splits/test.csv")

BATCH_SIZE = 32
NUM_WORKERS = 0


# ============================================================
# DATASETS
# ============================================================

def create_datasets():

    train_dataset = VehicleFraudDataset(
        TRAIN_CSV,
        transform=train_transform,
    )

    val_dataset = VehicleFraudDataset(
        VAL_CSV,
        transform=eval_transform,
    )

    test_dataset = VehicleFraudDataset(
        TEST_CSV,
        transform=eval_transform,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders():

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_datasets()

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# PIPELINE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LEAKAGE-SAFE DATA PIPELINE")
    print("=" * 60)

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders()

    print()
    print("Dataset sizes:")

    print(
        f"Train      : "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation : "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test       : "
        f"{len(test_loader.dataset)}"
    )

    # --------------------------------------------------------
    # Check one batch
    # --------------------------------------------------------

    images, labels = next(
        iter(train_loader)
    )

    print()
    print("First training batch:")

    print(
        f"Images shape : "
        f"{images.shape}"
    )

    print(
        f"Labels shape : "
        f"{labels.shape}"
    )

    print()
    print("Image statistics:")

    print(
        f"Min value : "
        f"{images.min().item():.4f}"
    )

    print(
        f"Max value : "
        f"{images.max().item():.4f}"
    )

    print()
    print("Labels in first batch:")

    print(labels)

    print()
    print(
        "Fraud samples : "
        f"{(labels == 1).sum().item()}"
    )

    print(
        "Non-Fraud samples : "
        f"{(labels == 0).sum().item()}"
    )

    print()
    print("=" * 60)
    print("LEAKAGE-SAFE DATA PIPELINE TEST PASSED")
    print("=" * 60)