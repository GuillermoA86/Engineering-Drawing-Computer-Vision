import json
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model import build_model
from preprocessing import preprocess_pil

st.set_page_config(page_title="Engineering Drawing CV", page_icon="⚙️")

st.title("Engineering Drawing Symbol Recognition")
st.write(
    "Upload a cropped engineering symbol from a P&ID and classify it "
    "with the trained computer-vision model."
)

checkpoint_path = "models/best_model.pt"

@st.cache_resource
def load_model():
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model = build_model(len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["image_size"], idx_to_class

uploaded = st.file_uploader(
    "Upload a symbol image",
    type=["png", "jpg", "jpeg", "bmp"],
)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Input engineering symbol", use_container_width=True)

    if not torch.jit.is_scripting():
        try:
            model, size, idx_to_class = load_model()
            processed = preprocess_pil(image, size)
            x = transforms.ToTensor()(processed)
            x = transforms.Normalize([0.5], [0.5])(x).unsqueeze(0)

            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0]

            values, indices = probs.topk(min(5, len(probs)))

            st.subheader("Prediction")
            st.metric(
                "Predicted symbol",
                idx_to_class[indices[0].item()],
                f"{float(values[0])*100:.1f}% confidence",
            )

            rows = [
                {
                    "class": idx_to_class[i.item()],
                    "probability": float(v),
                }
                for v, i in zip(values, indices)
            ]
            st.dataframe(rows, use_container_width=True)
        except FileNotFoundError:
            st.warning(
                "No trained checkpoint found. Run the training pipeline first."
            )
