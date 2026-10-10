import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import EngineeringDrawingDataset
from src.model import build_model


CSV_PATH = "data/raw/Eng_Diagrams/data/Symbols_pixel.csv"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            predictions = logits.argmax(dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return accuracy, balanced_acc, macro_f1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/resnet18.yaml",
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
    # Configuration
    # ---------------------------------------------------------

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["seed"])

    # ---------------------------------------------------------
    # Load manifest
    # ---------------------------------------------------------

    manifest = pd.read_csv(args.manifest)

    classes = sorted(
        manifest["label"].astype(str).unique()
    )

    class_to_idx = {
        label: idx
        for idx, label in enumerate(classes)
    }

    print(f"Total samples: {len(manifest)}")
    print(f"Number of classes: {len(classes)}")

    # ---------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------

    train_ds = EngineeringDrawingDataset(
        manifest_path=args.manifest,
        csv_path=args.csv,
        split="train",
        label_to_index=class_to_idx,
        augment=True,
    )

    val_ds = EngineeringDrawingDataset(
        manifest_path=args.manifest,
        csv_path=args.csv,
        split="validation",
        label_to_index=class_to_idx,
        augment=False,
    )

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    # ---------------------------------------------------------
    # Data loaders
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = build_model(
        len(classes),
        cfg["pretrained"],
    ).to(device)

    # ---------------------------------------------------------
    # Class-weighted loss
    # ---------------------------------------------------------

    weights = None

    if cfg.get("class_weighted_loss", True):

        train_labels = (
            train_ds.manifest["label"]
            .astype(str)
            .map(class_to_idx)
            .to_numpy()
        )

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(classes)),
            y=train_labels,
        )

        weights = torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )

        print("Using class-weighted CrossEntropyLoss.")

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    Path("models").mkdir(
        exist_ok=True
    )

    best_macro_f1 = -1.0

    history = []

    for epoch in range(cfg["epochs"]):

        model.train()

        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg['epochs']}",
        )

        for x, y in progress:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * x.size(0)
            )

        train_loss = (
            running_loss
            / max(len(train_ds), 1)
        )

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        val_accuracy, val_balanced_accuracy, val_macro_f1 = (
            evaluate_model(
                model,
                val_loader,
                device,
            )
        )

        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f} "
            f"val_balanced_accuracy={val_balanced_accuracy:.4f} "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_accuracy": val_accuracy,
                "val_balanced_accuracy": val_balanced_accuracy,
                "val_macro_f1": val_macro_f1,
            }
        )

        # -----------------------------------------------------
        # Save best model based on macro F1
        # -----------------------------------------------------

        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = val_macro_f1

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "image_size": cfg["image_size"],
                },
                "models/best_model.pt",
            )

            print(
                f"Saved best model. "
                f"Validation macro F1: {best_macro_f1:.4f}"
            )

    # ---------------------------------------------------------
    # Save training history
    # ---------------------------------------------------------

    Path("outputs").mkdir(
        exist_ok=True
    )

    with open(
        "outputs/training_history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    print(
        f"\nBest validation macro F1: "
        f"{best_macro_f1:.4f}"
    )


if __name__ == "__main__":
    main()