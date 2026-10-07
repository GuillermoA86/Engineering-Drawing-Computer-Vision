import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "raw" / "Eng_Diagrams"
URL = "https://github.com/heyad/Eng_Diagrams.git"

def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"Dataset repository already exists: {TARGET}")
        return
    subprocess.run(
        ["git", "clone", "--depth", "1", URL, str(TARGET)],
        check=True,
    )
    print(f"Downloaded dataset repository to {TARGET}")

if __name__ == "__main__":
    main()
