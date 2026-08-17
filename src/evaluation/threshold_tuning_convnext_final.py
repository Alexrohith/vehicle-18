from pathlib import Path
import json

import numpy as np
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from src.data.loader import create_dataloaders
from src.models.convnext import ConvNeXtClassifier


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = Path(
    "models/convnext_final_development_best.pth"
)

OUTPUT_PATH = Path(
    "artifacts/metrics/"
    "convnext_final_optimal_threshold.json"
)

# IMPORTANT:
# Threshold is selected ONLY using validation data.
THRESHOLDS = np.arange(
    0.05,
    0.96,
    0.01
)


# ============================================================
# HEADER
# ============================================================

def print_header():

    print("=" * 60)
    print("CONVNEXT FINAL VALIDATION THRESHOLD OPTIMIZATION")
    print("=" * 60)

    print()
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print(
        "Loading best development checkpoint..."
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

    return model


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

@torch.no_grad()
def get_validation_probabilities(
    model,
    val_loader,
):

    print()
    print(
        "Generating probabilities "
        "from VALIDATION set..."
    )

    all_labels = []
    all_probabilities = []

    for images, labels in val_loader:

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
# EVALUATE THRESHOLD
# ============================================================

def evaluate_threshold(
    labels,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

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

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    cm = confusion_matrix(
        labels,
        predictions,
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders()

    print()
    print(
        "Dataset:"
    )

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
        "🔒 TEST SET WILL NOT BE USED."
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # VALIDATION ONLY
    # --------------------------------------------------------

    labels, probabilities = (
        get_validation_probabilities(
            model,
            val_loader,
        )
    )

    print()
    print(
        f"Validation samples: "
        f"{len(labels)}"
    )

    print(
        f"Validation fraud: "
        f"{int(labels.sum())}"
    )

    # --------------------------------------------------------
    # THRESHOLD SEARCH
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SEARCHING VALIDATION THRESHOLDS")
    print("=" * 60)

    results = []

    for threshold in THRESHOLDS:

        result = evaluate_threshold(
            labels,
            probabilities,
            threshold,
        )

        results.append(result)

    # --------------------------------------------------------
    # SORT BY F1
    # --------------------------------------------------------

    results_sorted = sorted(
        results,
        key=lambda x: (
            x["f1"],
            x["recall"],
            x["precision"],
        ),
        reverse=True,
    )

    best = results_sorted[0]

    # --------------------------------------------------------
    # RANKING METRICS
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        labels,
        probabilities,
    )

    pr_auc = average_precision_score(
        labels,
        probabilities,
    )

    # --------------------------------------------------------
    # DISPLAY BEST
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("BEST VALIDATION THRESHOLD")
    print("=" * 60)

    print()
    print(
        f"Threshold : "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{best['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{best['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{pr_auc:.4f}"
    )

    print()
    print(
        "Confusion Matrix:"
    )

    print(
        np.array(
            best["confusion_matrix"]
        )
    )

    # --------------------------------------------------------
    # TOP 10
    # --------------------------------------------------------

    print()
    print(
        "Top 10 validation thresholds by F1:"
    )

    print()

    print(
        "Threshold   Precision   Recall      F1"
    )

    print("-" * 48)

    for result in results_sorted[:10]:

        print(
            f"{result['threshold']:<11.2f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1']:.4f}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {

        "model":
            "ConvNeXt-Tiny",

        "checkpoint":
            str(CHECKPOINT_PATH),

        "checkpoint_epoch":
            int(
                torch.load(
                    CHECKPOINT_PATH,
                    map_location="cpu",
                    weights_only=False,
                )["epoch"]
            ),

        "dataset":
            "data/final_splits",

        "validation_samples":
            int(len(labels)),

        "test_samples":
            int(len(test_loader.dataset)),

        "test_used":
            False,

        "threshold_selection_split":
            "validation",

        "selected_metric":
            "F1",

        "optimal_threshold":
            best["threshold"],

        "validation_roc_auc":
            float(roc_auc),

        "validation_pr_auc":
            float(pr_auc),

        "best_validation_result":
            best,

        "top_10_thresholds":
            results_sorted[:10],
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
        )

    print()
    print(
        "Threshold configuration saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 60)
    print(
        "VALIDATION THRESHOLD OPTIMIZATION COMPLETE"
    )
    print("=" * 60)

    print()
    print(
        "🔒 TEST SET WAS NOT USED."
    )

    print(
        "The threshold is now ready to be LOCKED."
    )


if __name__ == "__main__":
    main()