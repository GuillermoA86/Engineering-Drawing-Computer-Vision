from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "Eng_Diagrams"
OUT = ROOT / "data" / "processed" / "manifest.csv"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def infer_label(path: Path) -> str:
    # The SiED repository is expected to expose class information through
    # the dataset directory structure. If the upstream structure changes,
    # adapt this function or build a custom manifest.
    return path.parent.name

def main():
    if not DATASET.exists():
        raise FileNotFoundError("Run scripts/download_dataset.py first.")

    images = [
        p for p in DATASET.rglob("*")
        if p.suffix.lower() in EXTENSIONS
    ]

    records = []
    for path in images:
        try:
            with Image.open(path) as im:
                width, height = im.size
        except Exception:
            continue

        label = infer_label(path)
        if label.lower() in {"data", "images", "train", "test", "validation", "val"}:
            continue

        records.append({
            "path": str(path.resolve()),
            "label": label,
            "width": width,
            "height": height,
        })

    if not records:
        raise RuntimeError(
            "No class-folder images were found. Inspect the upstream repository "
            "and adapt infer_label() to its current annotation format."
        )

    df = pd.DataFrame(records).drop_duplicates("path")

    # Reproducible stratified split.
    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=42,
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp["label"],
        random_state=42,
    )

    train["split"] = "train"
    val["split"] = "validation"
    test["split"] = "test"

    result = pd.concat([train, val, test], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)

    print(f"Manifest written to {OUT}")
    print(result.groupby(["split", "label"]).size().unstack(fill_value=0))

if __name__ == "__main__":
    main()
