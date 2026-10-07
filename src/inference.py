import argparse
import json
import torch
from PIL import Image

from model import build_model
from preprocessing import preprocess_pil
from torchvision import transforms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    image = Image.open(args.image)
    image = preprocess_pil(image, ckpt["image_size"])
    x = transforms.ToTensor()(image)
    x = transforms.Normalize([0.5], [0.5])(x).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    values, indices = probs.topk(min(args.top_k, len(probs)))

    result = {
        "predicted_class": idx_to_class[indices[0].item()],
        "confidence": float(values[0]),
        "top_k": [
            {
                "class": idx_to_class[i.item()],
                "probability": float(v),
            }
            for v, i in zip(values, indices)
        ],
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
