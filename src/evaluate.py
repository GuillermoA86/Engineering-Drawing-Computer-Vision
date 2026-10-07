import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, top_k_accuracy_score
)
from torch.utils.data import DataLoader

from dataset import EngineeringSymbolDataset
from model import build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    ds = EngineeringSymbolDataset(
        args.manifest, "test", ckpt["image_size"], class_to_idx, train=False
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    model = build_model(len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(prob.argmax(1).numpy())
            y_prob.append(prob.numpy())

    import numpy as np
    y_prob = np.concatenate(y_prob)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "top3_accuracy": top_k_accuracy_score(
            y_true, y_prob, k=min(3, len(class_to_idx)),
            labels=list(range(len(class_to_idx)))
        ),
    }

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(
        y_true, y_pred,
        target_names=[idx_to_class[i] for i in range(len(idx_to_class))],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv("outputs/classification_report.csv")

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm).to_csv("outputs/confusion_matrix.csv")

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
