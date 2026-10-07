import io
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from model import build_model
from preprocessing import preprocess_pil

app = FastAPI(title="Engineering Drawing Symbol Classifier")

CHECKPOINT = "models/best_model.pt"

def load_model():
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model = build_model(len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["image_size"], idx_to_class

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model, size, idx_to_class = load_model()
    image = Image.open(io.BytesIO(await file.read()))
    image = preprocess_pil(image, size)
    x = transforms.ToTensor()(image)
    x = transforms.Normalize([0.5], [0.5])(x).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    values, indices = probs.topk(min(3, len(probs)))
    return {
        "predicted_class": idx_to_class[indices[0].item()],
        "confidence": float(values[0]),
        "top_k": [
            {"class": idx_to_class[i.item()], "probability": float(v)}
            for v, i in zip(values, indices)
        ],
    }
