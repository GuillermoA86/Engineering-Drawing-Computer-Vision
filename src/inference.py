import argparse
import json

from PIL import Image

from src.predict import load_model, predict_image


def main():
    parser = argparse.ArgumentParser(
        description="Run engineering symbol classification inference."
    )

    parser.add_argument(
        "--checkpoint",
        default="models/best_model.pt",
        help="Path to the trained model checkpoint.",
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to return.",
    )

    args = parser.parse_args()

    model, image_size, idx_to_class = load_model(
        args.checkpoint
    )

    image = Image.open(
        args.image
    )

    result = predict_image(
        image=image,
        model=model,
        image_size=image_size,
        idx_to_class=idx_to_class,
        top_k=args.top_k,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()