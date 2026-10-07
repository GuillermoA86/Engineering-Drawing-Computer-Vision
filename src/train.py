import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.utils.class_weight import compute_class_weight
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EngineeringSymbolDataset
from model import build_model

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet18.yaml")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["seed"])

    manifest = pd.read_csv(args.manifest)
    classes = sorted(manifest["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_ds = EngineeringSymbolDataset(
        args.manifest, "train", cfg["image_size"], class_to_idx, train=True
    )
    val_ds = EngineeringSymbolDataset(
        args.manifest, "validation", cfg["image_size"], class_to_idx, train=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"]
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(classes), cfg["pretrained"]).to(device)

    weights = None
    if cfg.get("class_weighted_loss", True):
        y = train_ds.df["label"].map(class_to_idx).to_numpy()
        values = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y)
        weights = torch.tensor(values, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    best = 0.0
    Path("models").mkdir(exist_ok=True)

    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()

        val_acc = correct / max(total, 1)
        print(f"epoch={epoch+1} train_loss={running/max(len(train_ds),1):.4f} val_accuracy={val_acc:.4f}")

        if val_acc > best:
            best = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "image_size": cfg["image_size"],
                },
                "models/best_model.pt",
            )

    print(f"Best validation accuracy: {best:.4f}")

if __name__ == "__main__":
    main()
