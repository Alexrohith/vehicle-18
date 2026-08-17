import torch
from torchvision import transforms


IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# STANDARD TRAIN AUGMENTATION
# ============================================================

train_transform_regularized = transforms.Compose([

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.85, 1.0),
        ratio=(0.90, 1.10),
        antialias=True
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.15,
        hue=0.02
    ),

    # Small blur improves robustness to different
    # image quality/camera conditions.
    transforms.RandomApply(
        [
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.0)
            )
        ],
        p=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),

    # Random erasing acts as regularization.
    # Keep probability low because damage location
    # can be important.
    transforms.RandomErasing(
        p=0.10,
        scale=(0.01, 0.05),
        ratio=(0.5, 2.0),
        value="random"
    )
])


# ============================================================
# VALIDATION / TEST
# ============================================================

eval_transform_regularized = transforms.Compose([

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