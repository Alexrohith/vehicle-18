from pathlib import Path
import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

from src.data.loader import create_dataloaders
from src.models.convnext import ConvNeXtClassifier


# ============================================================
# CONFIG — EVERYTHING IS NOW LOCKED
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = Path(
    "models/convnext_final_development_best.pth"
)

THRESHOLD_PATH = Path(
    "artifacts/metrics/"
    "convnext_final_optimal_threshold.json"
)

OUTPUT_PATH = Path(
    "artifacts/metrics/"
    "convnext_final_test_metrics.json"
)


# ============================================================
# HEADER
# ============================================================

def print_header():

    print("=" * 60)
    print("CONVNEXT-TINY FINAL HELD-OUT TEST EVALUATION")
    print("=" * 60)

    print()
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )


# ============================================================
# LOAD LOCKED THRESHOLD
# ============================================================

def load_locked_threshold():

    print()
    print("Loading LOCKED threshold...")

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    threshold = float(
        config["optimal_threshold"]
    )

    print(
        f"Locked threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{config['best_validation_result']['f1']:.4f}"
    )

    print()
    print(
        "🔒 Threshold will NOT be changed."
    )

    return threshold


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print(
        "Loading LOCKED development checkpoint..."
    )

    model = ConvNeXtClassifier()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    stored_metrics = checkpoint.get(
        "validation_metrics",
        {}
    )

    print(
        f"Stored validation F1: "
        f"{stored_metrics.get('f1', 'N/A')}"
    )

    return model, checkpoint


# ============================================================
# TEST INFERENCE
# ============================================================

@torch.no_grad()
def run_test_inference(
    model,
    test_loader,
):

    print()
    print(
        "Running inference on FINAL TEST set..."
    )

    all_labels = []
    all_probabilities = []

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(images)

        logits = logits.squeeze(1)

        probabilities = torch.sigmoid(
            logits
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

    labels = np.asarray(
        all_labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        all_probabilities,
        dtype=np.float64,
    )

    return labels, probabilities


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    # --------------------------------------------------------
    # LOAD LOCKED THRESHOLD
    # --------------------------------------------------------

    threshold = load_locked_threshold()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders()

    print()
    print("DATASET")

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
        "🔒 ONLY THE HELD-OUT TEST SET "
        "WILL BE EVALUATED."
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model, checkpoint = load_model()

    # --------------------------------------------------------
    # TEST INFERENCE
    # --------------------------------------------------------

    labels, probabilities = (
        run_test_inference(
            model,
            test_loader,
        )
    )

    # --------------------------------------------------------
    # APPLY LOCKED THRESHOLD
    # --------------------------------------------------------

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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

    report = classification_report(
        labels,
        predictions,
        target_names=[
            "Non-Fraud",
            "Fraud",
        ],
        digits=4,
        zero_division=0,
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX VALUES
    # --------------------------------------------------------

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL CONVNEXT-TINY TEST RESULTS")
    print("=" * 60)

    print()
    print(
        f"Test samples : {len(labels)}"
    )

    print(
        f"Accuracy     : {accuracy:.4f}"
    )

    print(
        f"Precision    : {precision:.4f}"
    )

    print(
        f"Recall       : {recall:.4f}"
    )

    print(
        f"F1 Score     : {f1:.4f}"
    )

    print(
        f"ROC-AUC      : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC       : {pr_auc:.4f}"
    )

    print(
        f"Threshold    : {threshold:.2f}"
    )

    print()
    print("Confusion Matrix:")

    print(cm)

    print()
    print("Confusion Matrix Breakdown:")

    print(
        f"True Negatives  : {tn}"
    )

    print(
        f"False Positives : {fp}"
    )

    print(
        f"False Negatives : {fn}"
    )

    print(
        f"True Positives   : {tp}"
    )

    print()
    print("Classification Report:")
    print(report)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {

        "model":
            "ConvNeXt-Tiny",

        "checkpoint":
            str(CHECKPOINT_PATH),

        "checkpoint_epoch":
            int(checkpoint["epoch"]),

        "threshold_config":
            str(THRESHOLD_PATH),

        "threshold":
            float(threshold),

        "dataset":
            "data/final_splits",

        "train_samples":
            int(len(train_loader.dataset)),

        "validation_samples":
            int(len(val_loader.dataset)),

        "test_samples":
            int(len(test_loader.dataset)),

        "test_used_for_model_selection":
            False,

        "test_used_for_threshold_selection":
            False,

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            float(roc_auc),

        "pr_auc":
            float(pr_auc),

        "true_negatives":
            int(tn),

        "false_positives":
            int(fp),

        "false_negatives":
            int(fn),

        "true_positives":
            int(tp),

        "confusion_matrix":
            cm.tolist(),

        "classification_report":
            report,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 60)
    print(
        "FINAL TEST EVALUATION COMPLETE ✅"
    )
    print("=" * 60)

    print()
    print(
        "🔒 MODEL, CHECKPOINT AND THRESHOLD "
        "WERE NOT CHANGED."
    )


if __name__ == "__main__":
    main()