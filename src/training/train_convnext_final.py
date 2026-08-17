from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from tqdm import tqdm

from src.data.loader import create_dataloaders
from src.models.convnext import ConvNeXtClassifier


# ============================================================
# CONFIG
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 15

STAGE1_LR = 1e-3
STAGE2_LR = 1e-5

WEIGHT_DECAY = 1e-4

CHECKPOINT_PATH = Path(
    "models/convnext_final_development_best.pth"
)

METRICS_PATH = Path(
    "artifacts/metrics/"
    "convnext_final_development_metrics.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

def print_device():

    print("=" * 60)
    print("FINAL LEAKAGE-SAFE CONVNEXT-TINY")
    print("=" * 60)

    print()
    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )


# ============================================================
# MODEL
# ============================================================

def create_model():

    model = ConvNeXtClassifier()

    return model.to(DEVICE)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities,
    threshold=0.5,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities,
    )

    pr_auc = average_precision_score(
        labels,
        probabilities,
    )

    cm = confusion_matrix(
        labels,
        predictions,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()

    running_loss = 0.0

    all_labels = []
    all_probs = []

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for images, labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        logits = logits.squeeze(1)

        loss = criterion(
            logits,
            labels.float(),
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        probabilities = torch.sigmoid(
            logits
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_probs.extend(
            probabilities.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_probs),
    )

    return epoch_loss, metrics


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_probs = []

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(images)

        logits = logits.squeeze(1)

        loss = criterion(
            logits,
            labels.float(),
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        probabilities = torch.sigmoid(
            logits
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_probs.extend(
            probabilities.cpu().numpy()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_probs),
    )

    return epoch_loss, metrics


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
):

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "validation_metrics":
                metrics,
        },
        CHECKPOINT_PATH,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print_device()

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders()

    print()
    print("DATASET SIZES")

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

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Test loader will NOT be accessed "
        "during training."
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print(
        "Loading fresh ConvNeXt-Tiny..."
    )

    model = create_model()

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    best_epoch = -1
    best_metrics = None

    # ========================================================
    # STAGE 1
    # ========================================================

    print()
    print("=" * 60)
    print(
        "STAGE 1 — CLASSIFIER HEAD"
    )
    print("=" * 60)

    # Freeze everything first.
    for param in model.parameters():

        param.requires_grad = False

    # Try to locate classifier.
    classifier = None

    if hasattr(model, "classifier"):

        classifier = model.classifier

    elif hasattr(model, "model") and hasattr(
        model.model,
        "classifier",
    ):

        classifier = model.model.classifier

    if classifier is None:

        raise RuntimeError(
            "Could not locate ConvNeXt classifier."
        )

    for param in classifier.parameters():

        param.requires_grad = True

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print(
        f"Total parameters    : "
        f"{total:,}"
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=STAGE1_LR,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(
        1,
        STAGE1_EPOCHS + 1,
    ):

        train_loss, train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
        )

        val_loss, val_metrics = (
            validate(
                model,
                val_loader,
                criterion,
            )
        )

        print()
        print(
            f"Epoch [{epoch}]"
        )

        print(
            f"Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss   : "
            f"{val_loss:.4f}"
        )

        print(
            f"Accuracy   : "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Precision  : "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"Recall     : "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"F1 Score   : "
            f"{val_metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC    : "
            f"{val_metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC     : "
            f"{val_metrics['pr_auc']:.4f}"
        )

        print(
            f"Learning Rate: "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )

        print()
        print("Confusion Matrix:")

        print(
            np.array(
                val_metrics[
                    "confusion_matrix"
                ]
            )
        )

        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]

            best_epoch = epoch

            best_metrics = val_metrics

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
            )

            print()
            print(
                "🔥 Best ConvNeXt model saved!"
            )

            print(
                f"Best F1: {best_f1:.4f}"
            )

    # ========================================================
    # STAGE 2
    # ========================================================

    print()
    print("=" * 60)
    print(
        "STAGE 2 — FINE-TUNE CONVNEXT-TINY"
    )
    print("=" * 60)

    # Unfreeze backbone.
    for param in model.parameters():

        param.requires_grad = True

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=STAGE2_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=STAGE2_EPOCHS,
    )

    for local_epoch in range(
        1,
        STAGE2_EPOCHS + 1,
    ):

        epoch = (
            STAGE1_EPOCHS
            + local_epoch
        )

        train_loss, train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
        )

        val_loss, val_metrics = (
            validate(
                model,
                val_loader,
                criterion,
            )
        )

        scheduler.step()

        print()
        print(
            f"Epoch [{epoch}]"
        )

        print(
            f"Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss   : "
            f"{val_loss:.4f}"
        )

        print(
            f"Accuracy   : "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Precision  : "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"Recall     : "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"F1 Score   : "
            f"{val_metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC    : "
            f"{val_metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC     : "
            f"{val_metrics['pr_auc']:.4f}"
        )

        print(
            f"Learning Rate: "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )

        print()
        print("Confusion Matrix:")

        print(
            np.array(
                val_metrics[
                    "confusion_matrix"
                ]
            )
        )

        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]

            best_epoch = epoch

            best_metrics = val_metrics

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
            )

            print()
            print(
                "🔥 Best ConvNeXt model saved!"
            )

            print(
                f"Best F1: {best_f1:.4f}"
            )

    # --------------------------------------------------------
    # SAVE DEVELOPMENT RESULTS
    # --------------------------------------------------------

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {

        "model":
            "ConvNeXt-Tiny",

        "dataset":
            "data/final_splits",

        "seed":
            SEED,

        "train_samples":
            len(train_loader.dataset),

        "validation_samples":
            len(val_loader.dataset),

        "test_samples":
            len(test_loader.dataset),

        "test_used_during_training":
            False,

        "test_used_for_selection":
            False,

        "best_epoch":
            best_epoch,

        "best_validation_f1":
            best_f1,

        "best_validation_metrics":
            best_metrics,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 60)
    print(
        "CONVNEXT DEVELOPMENT TRAINING COMPLETE"
    )
    print("=" * 60)

    print()
    print(
        f"Best Epoch: {best_epoch}"
    )

    print(
        f"Best Validation F1: "
        f"{best_f1:.4f}"
    )

    print()
    print(
        f"Model saved to:"
    )

    print(
        CHECKPOINT_PATH
    )

    print()
    print(
        "TEST SET WAS NOT USED."
    )

    print(
        "Do NOT evaluate test yet."
    )

    print(
        "Next step: threshold tuning "
        "on VALIDATION ONLY."
    )


if __name__ == "__main__":
    main()