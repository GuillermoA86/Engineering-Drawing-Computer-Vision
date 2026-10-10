import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.dataset import EngineeringDrawingDataset
from src.model import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data/processed/manifest.csv"
CSV_PATH = PROJECT_ROOT / "data/raw/Eng_Diagrams/data/Symbols_pixel.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "models/best_model.pt"


def main():
    # Load checkpoint.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    class_to_idx = checkpoint["class_to_idx"]

    idx_to_class = {
        value: key
        for key, value in class_to_idx.items()
    }

    # Create the test dataset using the exact same
    # dataset implementation used during evaluation.
    test_dataset = EngineeringDrawingDataset(
        manifest_path=MANIFEST_PATH,
        csv_path=CSV_PATH,
        split="test",
        label_to_index=class_to_idx,
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # Build model.
    model = build_model(
        num_classes=len(class_to_idx),
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.eval()

    predictions = []
    true_labels = []
    top3_correct = []

    with torch.no_grad():

        for images, targets in test_loader:

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predicted_indices = torch.argmax(
                probabilities,
                dim=1,
            )

            predictions.extend(
                predicted_indices.tolist()
            )

            true_labels.extend(
                targets.tolist()
            )

            k = min(
                3,
                probabilities.shape[1],
            )

            top3_indices = torch.topk(
                probabilities,
                k=k,
                dim=1,
            ).indices

            for target, top3 in zip(
                targets,
                top3_indices,
            ):
                top3_correct.append(
                    target.item() in top3.tolist()
                )

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
    )

    top3_accuracy = float(
        np.mean(top3_correct)
    )

    results = {
        "test_samples": len(test_dataset),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "top3_accuracy": top3_accuracy,
    }

    print(
        json.dumps(
            results,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()