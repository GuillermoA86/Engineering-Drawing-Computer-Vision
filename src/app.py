import streamlit as st
from PIL import Image

from predict import load_model, predict_image


st.set_page_config(
    page_title="Engineering Drawing CV",
    page_icon="⚙️",
    layout="centered",
)


st.title(
    "Engineering Drawing Symbol Recognition"
)

st.write(
    "Upload a cropped engineering symbol from a "
    "Piping & Instrumentation Diagram (P&ID) and "
    "classify it with a ResNet-18 computer-vision model."
)


CHECKPOINT_PATH = "models/best_model.pt"


@st.cache_resource
def get_model():
    return load_model(
        CHECKPOINT_PATH
    )


uploaded = st.file_uploader(
    "Upload an engineering symbol",
    type=[
        "png",
        "jpg",
        "jpeg",
        "bmp",
    ],
)


if uploaded:

    image = Image.open(
        uploaded
    )

    st.image(
        image,
        caption="Input engineering symbol",
        use_container_width=True,
    )

    try:

        model, image_size, idx_to_class = get_model()

        result = predict_image(
            image=image,
            model=model,
            image_size=image_size,
            idx_to_class=idx_to_class,
            top_k=3,
        )

        st.subheader(
            "Prediction"
        )

        st.metric(
            "Predicted symbol",
            result["predicted_class"],
            f"{result['confidence'] * 100:.2f}% confidence",
        )

        st.subheader(
            "Top 3 predictions"
        )

        for rank, item in enumerate(
            result["top_k"],
            start=1,
        ):

            st.write(
                f"**{rank}. {item['class']}**"
            )

            st.progress(
                item["probability"]
            )

            st.caption(
                f"{item['probability'] * 100:.2f}%"
            )

    except FileNotFoundError:

        st.error(
            "Model checkpoint not found. "
            "Please train the model first."
        )

    except Exception as exc:

        st.error(
            f"Prediction failed: {exc}"
        )