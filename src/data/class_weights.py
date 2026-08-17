import pandas as pd
import torch


TRAIN_CSV = "data/splits/train.csv"


def calculate_class_weights():

    df = pd.read_csv(TRAIN_CSV)

    counts = df["label"].value_counts().sort_index()

    total = len(df)
    num_classes = len(counts)

    weights = total / (num_classes * counts)

    weights = torch.tensor(
        weights.values,
        dtype=torch.float32
    )

    print("Class counts:")
    print(counts)

    print("\nClass weights:")
    print(weights)

    return weights


if __name__ == "__main__":

    calculate_class_weights()