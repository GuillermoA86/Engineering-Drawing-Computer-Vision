import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .preprocessing import preprocess_pil

class EngineeringSymbolDataset(Dataset):
    def __init__(self, manifest_path, split, image_size=128, class_to_idx=None, train=False):
        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

        if class_to_idx is None:
            classes = sorted(self.df["label"].unique())
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        ops = [
            transforms.Lambda(lambda im: preprocess_pil(im, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]

        if train:
            ops.insert(1, transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ))

        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"])
        image = self.transform(image)
        label = self.class_to_idx[row["label"]]
        return image, label
