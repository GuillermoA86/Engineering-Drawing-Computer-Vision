from pathlib import Path
from collections import Counter
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "Eng_Diagrams"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def main():
    if not DATASET.exists():
        raise FileNotFoundError("Run scripts/download_dataset.py first.")

    images = [p for p in DATASET.rglob("*") if p.suffix.lower() in EXTENSIONS]
    print(f"Images found: {len(images)}")

    sizes = Counter()
    parent_classes = Counter()

    for path in images:
        try:
            with Image.open(path) as im:
                sizes[im.size] += 1
        except Exception:
            continue
        parent_classes[path.parent.name] += 1

    print("\nMost common image dimensions:")
    for size, count in sizes.most_common(10):
        print(f"  {size}: {count}")

    print("\nCandidate parent-directory labels:")
    for label, count in parent_classes.most_common(30):
        print(f"  {label}: {count}")

if __name__ == "__main__":
    main()
