from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing import preprocess_image


class EngineeringDrawingDataset(Dataset):
    """
    Dataset for the SiED engineering-symbol dataset.

    Each sample contains:
    - 10,000 grayscale pixel values
    - 1 categorical label

    The 10,000 pixels are reshaped into a 100x100 image.
    """

    def __init__(
        self,
        manifest_path,
        csv_path,
        split,
        label_to_index=None,
        augment=False,
    ):
        self.manifest_path = Path(manifest_path)
        self.csv_path = Path(csv_path)
        self.split = split
        self.augment = augment

        self.manifest = pd.read_csv(self.manifest_path)
        self.manifest = self.manifest[
            self.manifest["split"] == split
        ].reset_index(drop=True)

        # Load the original CSV without assuming a header.
        self.data = pd.read_csv(
            self.csv_path,
            header=None,
        )

        if label_to_index is None:
            labels = sorted(self.manifest["label"].unique())
            self.label_to_index = {
                label: idx
                for idx, label in enumerate(labels)
            }
        else:
            self.label_to_index = label_to_index

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]

        sample_id = int(row["sample_id"])
        label = row["label"]

        # Extract the flattened 100x100 image.
        pixels = self.data.iloc[
            sample_id,
            :10000,
        ].to_numpy(dtype="float32")

        image = pixels.reshape(100, 100)

        image = preprocess_image(
            image,
            augment=self.augment,
        )

        target = self.label_to_index[label]

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(target, dtype=torch.long),
        )