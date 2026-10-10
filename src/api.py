import io

from fastapi import FastAPI, File, UploadFile
from PIL import Image

try:
    from src.predict import load_model, predict_image
except ModuleNotFoundError:
    from predict import load_model, predict_image


app = FastAPI(
    title="Engineering Drawing Symbol Classifier",
    version="1.0.0",
)


CHECKPOINT_PATH = "models/best_model.pt"


model, image_size, idx_to_class = load_model(
    CHECKPOINT_PATH
)


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    result = predict_image(
        image=image,
        model=model,
        image_size=image_size,
        idx_to_class=idx_to_class,
        top_k=3,
    )

    return result