from pathlib import Path

import pandas as pd
import numpy as np


DATA_PATH = Path("data/raw/Eng_Diagrams/data/Symbols_pixel.csv")
OUTPUT_PATH = Path("data/processed/manifest.csv")

RANDOM_STATE = 42

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

MIN_SAMPLES_FOR_SPLIT = 3


def split_class_samples(group):
    """
    Split samples from one class into train/validation/test.

    Very small classes are kept entirely in training.
    Larger classes receive approximately an 80/10/10 split.
    """

    group = group.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    n = len(group)

    if n < MIN_SAMPLES_FOR_SPLIT:
        group["split"] = "train"
        return group

    # At least one sample for validation and test.
    n_test = max(1, round(n * TEST_RATIO))
    n_val = max(1, round(n * VALIDATION_RATIO))

    # Make sure train retains the majority of the samples.
    n_train = n - n_val - n_test

    # For very small classes, protect the training split.
    if n_train < 1:
        n_train = 1
        remaining = n - n_train

        if remaining >= 2:
            n_val = 1
            n_test = remaining - n_val
        else:
            n_val = remaining
            n_test = 0

    train = group.iloc[:n_train].copy()
    validation = group.iloc[
        n_train:n_train + n_val
    ].copy()

    test = group.iloc[
        n_train + n_val:
    ].copy()

    train["split"] = "train"
    validation["split"] = "validation"

    if len(test) > 0:
        test["split"] = "test"

    return pd.concat(
        [train, validation, test],
        ignore_index=True,
    )


def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print(
        f"Loading dataset: {DATA_PATH}"
    )

    # The original SiED CSV has no header.
    df = pd.read_csv(
        DATA_PATH,
        header=None,
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    expected_columns = 10001

    if df.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns "
            f"(10,000 pixels + 1 label), "
            f"but found {df.shape[1]}."
        )

    # Last column = class label.
    df["label"] = (
        df.iloc[:, 10000]
        .astype(str)
        .str.strip()
    )

    # Stable sample identifier.
    df["sample_id"] = np.arange(
        len(df)
    )

    metadata = df[
        ["sample_id", "label"]
    ].copy()

    print(
        f"Number of samples: "
        f"{len(metadata)}"
    )

    print(
        f"Number of classes: "
        f"{metadata['label'].nunique()}"
    )

    print("\nClass distribution:")

    print(
        metadata["label"]
        .value_counts()
        .to_string()
    )

    # ------------------------------------------------------------------
    # Split independently inside every class.
    # ------------------------------------------------------------------

    split_frames = []

    for label, group in metadata.groupby(
        "label",
        sort=True,
    ):

        count = len(group)

        if count < MIN_SAMPLES_FOR_SPLIT:
            print(
                f"\nRare class -> train only: "
                f"{label} ({count} sample(s))"
            )

        split_group = split_class_samples(
            group
        )

        split_frames.append(
            split_group
        )

    manifest = pd.concat(
        split_frames,
        ignore_index=True,
    )

    manifest = (
        manifest
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Validation checks.
    # ------------------------------------------------------------------

    if len(manifest) != len(metadata):
        raise RuntimeError(
            "Manifest sample count does not "
            "match the original dataset."
        )

    if manifest["sample_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate sample IDs detected."
        )

    if manifest["label"].nunique() != 39:
        raise RuntimeError(
            "Expected exactly 39 classes."
        )

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\n===================================="
    )
    print(
        "Manifest created successfully."
    )
    print(
        "===================================="
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print(
        f"Total samples: {len(manifest)}"
    )

    print("\nSplit distribution:")

    print(
        manifest["split"]
        .value_counts()
        .to_string()
    )

    print("\nClass distribution by split:")

    split_table = pd.crosstab(
        manifest["label"],
        manifest["split"],
    )

    print(
        split_table.to_string()
    )


if __name__ == "__main__":
    main()