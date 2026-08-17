from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from src.models.convnext import ConvNeXtClassifier


# ============================================================
# CONFIG — LOCKED FINAL MODEL
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = Path(
    "models/convnext_final_development_best.pth"
)

TEST_CSV = Path(
    "data/final_splits/test.csv"
)

THRESHOLD = 0.30

OUTPUT_DIR = Path(
    "artifacts/convnext_error_analysis"
)

OUTPUT_CSV = OUTPUT_DIR / "test_predictions.csv"

MISSED_FRAUD_DIR = OUTPUT_DIR / "missed_fraud"
FALSE_ALARM_DIR = OUTPUT_DIR / "false_alarms"


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MISSED_FRAUD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FALSE_ALARM_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("CONVNEXT FINAL ERROR ANALYSIS")
print("=" * 60)

print()
print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}")
print(f"Test CSV: {TEST_CSV}")
print(f"Locked threshold: {THRESHOLD}")

model = ConvNeXtClassifier()

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print(
    f"Checkpoint epoch: {checkpoint['epoch']}"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

df = pd.read_csv(TEST_CSV)

print()
print(f"Test samples: {len(df)}")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

from torchvision import transforms

IMAGE_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]

transform = transforms.Compose([
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
# INFERENCE
# ============================================================

print()
print("Running inference...")

results = []

with torch.no_grad():

    for index, row in df.iterrows():

        image_path = Path(row["path"])
        true_label = int(row["label"])

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            tensor = transform(
                image
            ).unsqueeze(0).to(DEVICE)

            logits = model(tensor)

            probability = torch.sigmoid(
                logits.squeeze()
            ).item()

            prediction = int(
                probability >= THRESHOLD
            )

            results.append({
                "index": index,
                "path": str(image_path),
                "true_label": true_label,
                "fraud_probability": probability,
                "prediction": prediction,
                "correct": (
                    prediction == true_label
                )
            })

        except Exception as e:

            print(
                f"ERROR: {image_path}"
            )

            print(e)


results_df = pd.DataFrame(results)


# ============================================================
# CLASSIFY ERRORS
# ============================================================

results_df["error_type"] = "correct"

results_df.loc[
    (results_df["true_label"] == 1) &
    (results_df["prediction"] == 0),
    "error_type"
] = "missed_fraud"

results_df.loc[
    (results_df["true_label"] == 0) &
    (results_df["prediction"] == 1),
    "error_type"
] = "false_alarm"


# ============================================================
# SUMMARY
# ============================================================

missed_fraud = results_df[
    results_df["error_type"] ==
    "missed_fraud"
]

false_alarms = results_df[
    results_df["error_type"] ==
    "false_alarm"
]

correct_fraud = results_df[
    (results_df["true_label"] == 1) &
    (results_df["prediction"] == 1)
]

correct_nonfraud = results_df[
    (results_df["true_label"] == 0) &
    (results_df["prediction"] == 0)
]


print()
print("=" * 60)
print("ERROR SUMMARY")
print("=" * 60)

print()
print(
    f"Total test images       : "
    f"{len(results_df)}"
)

print(
    f"Correct predictions     : "
    f"{results_df['correct'].sum()}"
)

print(
    f"Missed fraud            : "
    f"{len(missed_fraud)}"
)

print(
    f"False alarms            : "
    f"{len(false_alarms)}"
)

print(
    f"Correct fraud           : "
    f"{len(correct_fraud)}"
)

print(
    f"Correct non-fraud       : "
    f"{len(correct_nonfraud)}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print(
    f"Predictions saved to:"
)
print(OUTPUT_CSV)


# ============================================================
# SAVE ERROR LISTS
# ============================================================

missed_fraud.to_csv(
    OUTPUT_DIR / "missed_fraud.csv",
    index=False
)

false_alarms.to_csv(
    OUTPUT_DIR / "false_alarms.csv",
    index=False
)


# ============================================================
# DISPLAY MOST CONFIDENT ERRORS
# ============================================================

print()
print("=" * 60)
print("MISSED FRAUD — MOST CONFIDENT")
print("=" * 60)

if len(missed_fraud) > 0:

    print(
        missed_fraud.sort_values(
            "fraud_probability",
            ascending=True
        )[
            [
                "path",
                "true_label",
                "fraud_probability"
            ]
        ].head(13).to_string(
            index=False
        )
    )


print()
print("=" * 60)
print("FALSE ALARMS — MOST CONFIDENT")
print("=" * 60)

if len(false_alarms) > 0:

    print(
        false_alarms.sort_values(
            "fraud_probability",
            ascending=False
        )[
            [
                "path",
                "true_label",
                "fraud_probability"
            ]
        ].head(10).to_string(
            index=False
        )
    )


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

print()
print("=" * 60)
print("PROBABILITY SUMMARY")
print("=" * 60)

print()

print(
    "Fraud probabilities for TRUE FRAUD:"
)

print(
    results_df[
        results_df["true_label"] == 1
    ]["fraud_probability"].describe()
)


print()
print(
    "Fraud probabilities for TRUE NON-FRAUD:"
)

print(
    results_df[
        results_df["true_label"] == 0
    ]["fraud_probability"].describe()
)


print()
print("=" * 60)
print("ERROR ANALYSIS COMPLETE")
print("=" * 60)

print()
print(
    "IMPORTANT:"
)

print(
    "The model checkpoint and threshold "
    "were NOT changed."
)

print(
    "Test predictions are for analysis only."
)