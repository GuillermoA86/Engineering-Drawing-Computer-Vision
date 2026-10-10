from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from src.model import build_model
    from src.preprocessing import preprocess_image
except ModuleNotFoundError:
    from model import build_model
    from preprocessing import preprocess_image


DEFAULT_CHECKPOINT = Path("models/best_model.pt")


def load_model(checkpoint_path=DEFAULT_CHECKPOINT):
    """
    Load the trained engineering-symbol classifier.
    """
    checkpoint_path = Path(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    class_to_idx = checkpoint["class_to_idx"]

    idx_to_class = {
        value: key
        for key, value in class_to_idx.items()
    }

    image_size = checkpoint.get(
        "image_size",
        224,
    )

    model = build_model(
        num_classes=len(class_to_idx),
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.eval()

    return model, image_size, idx_to_class


def predict_image(
    image,
    model,
    image_size,
    idx_to_class,
    top_k=3,
):
    """
    Predict the engineering-symbol class for a PIL image.
    """

    # Convert image to grayscale.
    image = image.convert("L")

    # Convert PIL image to NumPy array.
    image_array = np.array(image)

    # Apply the exact preprocessing used during training.
    processed = preprocess_image(
        image_array,
        augment=False,
        image_size=image_size,
    )

    # Convert to PyTorch tensor.
    x = torch.tensor(
        processed,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(
            model(x),
            dim=1,
        )[0]

    k = min(
        top_k,
        len(probabilities),
    )

    values, indices = torch.topk(
        probabilities,
        k,
    )

    return {
        "predicted_class": idx_to_class[
            indices[0].item()
        ],
        "confidence": float(
            values[0]
        ),
        "top_k": [
            {
                "class": idx_to_class[
                    index.item()
                ],
                "probability": float(
                    value
                ),
            }
            for value, index in zip(
                values,
                indices,
            )
        ],
    }
