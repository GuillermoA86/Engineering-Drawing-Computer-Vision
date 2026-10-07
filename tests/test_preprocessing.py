from PIL import Image
from src.preprocessing import preprocess_pil

def test_preprocess_shape():
    image = Image.new("L", (100, 100), 255)
    result = preprocess_pil(image, 128)
    assert result.size == (128, 128)
    assert result.mode == "L"
