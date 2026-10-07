from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2

def preprocess_pil(image: Image.Image, size: int = 128) -> Image.Image:
    image = image.convert("L")
    image = ImageOps.autocontrast(image)

    arr = np.array(image)
    arr = cv2.GaussianBlur(arr, (3, 3), 0)

    # Preserve line geometry while improving contrast.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)

    image = Image.fromarray(arr)
    return image.resize((size, size), Image.Resampling.BILINEAR)
