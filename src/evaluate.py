import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader

from src.dataset import EngineeringDrawingDataset
from src.model import build_model


CSV_PATH = "data/raw/Eng_Diagrams/data/Symbols_pixel.csv"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="models/best_model.pt",
    )

    parser.add_argument(
        "--manifest",
        default="data/processed/manifest.csv",
    )

    parser.add_argument(
        "--csv",
        default=CSV_PATH,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    class_to_idx = checkpoint[
        "class_to_idx"
    ]

    idx_to_class = {
        index: label
        for label, index in class_to_idx.items()
    }

    # ---------------------------------------------------------
    # Test dataset
    # ---------------------------------------------------------

    test_ds = EngineeringDrawingDataset(
        manifest_path=args.manifest,
        csv_path=args.csv,
        split="test",
        label_to_index=class_to_idx,
        augment=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Test samples: {len(test_ds)}"
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = build_model(
        len(class_to_idx),
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.eval()

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():

        for x, y in test_loader:

            logits = model(x)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = probabilities.argmax(
                dim=1
            )

            y_true.extend(
                y.numpy()
            )

            y_pred.extend(
                predictions.numpy()
            )

            y_prob.append(
                probabilities.numpy()
            )

    y_prob = np.concatenate(
        y_prob,
        axis=0,
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    metrics = {

        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "top3_accuracy": top_k_accuracy_score(
            y_true,
            y_prob,
            k=min(
                3,
                len(class_to_idx),
            ),
            labels=list(
                range(
                    len(class_to_idx)
                )
            ),
        ),
    }

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------

    Path("outputs").mkdir(
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Metrics JSON
    # ---------------------------------------------------------

    with open(
        "outputs/metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(
                len(class_to_idx)
            )
        ),
        target_names=[
            idx_to_class[i]
            for i in range(
                len(class_to_idx)
            )
        ],
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(
        report
    ).T.to_csv(
        "outputs/classification_report.csv"
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(
                len(class_to_idx)
            )
        ),
    )

    pd.DataFrame(
        cm,
        index=[
            idx_to_class[i]
            for i in range(
                len(class_to_idx)
            )
        ],
        columns=[
            idx_to_class[i]
            for i in range(
                len(class_to_idx)
            )
        ],
    ).to_csv(
        "outputs/confusion_matrix.csv"
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()