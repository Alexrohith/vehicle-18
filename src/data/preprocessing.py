from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset

from torchvision import transforms


# Allow loading of slightly truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 224

# ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# TRAIN TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    # Preserve most of the original vehicle image
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.90, 1.0),
        ratio=(0.95, 1.05),
        antialias=True
    ),

    # Cars can naturally appear from either horizontal direction
    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    # Small viewpoint variation
    transforms.RandomRotation(
        degrees=8
    ),

    # Lighting variation
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02
    ),

    # Convert PIL → Tensor
    transforms.ToTensor(),

    # ImageNet normalization for pretrained models
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# VALIDATION / TEST TRANSFORMS
# ============================================================

eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        antialias=True
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# DATASET
# ============================================================

class VehicleFraudDataset(Dataset):

    def __init__(
        self,
        csv_file,
        transform=None
    ):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = row["path"]
        label = int(row["label"])

        try:

            image = Image.open(image_path).convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"Could not load image: {image_path}"
            ) from e

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(
            label,
            dtype=torch.long
        )


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def get_train_dataset():

    return VehicleFraudDataset(
        "data/splits/train.csv",
        transform=train_transform
    )


def get_validation_dataset():

    return VehicleFraudDataset(
        "data/splits/val.csv",
        transform=eval_transform
    )


def get_test_dataset():

    return VehicleFraudDataset(
        "data/splits/test.csv",
        transform=eval_transform
    )