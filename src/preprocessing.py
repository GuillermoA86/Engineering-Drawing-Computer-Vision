import cv2
import numpy as np


def preprocess_image(
    image,
    augment=False,
    image_size=224,
):
    """
    Preprocess a grayscale engineering symbol.

    Parameters
    ----------
    image : np.ndarray
        100x100 grayscale image.
    augment : bool
        Whether to apply lightweight augmentation.
    image_size : int
        Final image size for ResNet-18.

    Returns
    -------
    np.ndarray
        Tensor-ready image with shape (1, H, W).
    """

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    # Normalize pixel values to [0, 255].
    image_min = image.min()
    image_max = image.max()

    if image_max > image_min:
        image = (
            (image - image_min)
            / (image_max - image_min)
            * 255.0
        )
    else:
        image = np.zeros_like(image)

    image = image.astype(np.uint8)

    # Improve local contrast.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    image = clahe.apply(image)

    # Optional augmentation.
    if augment:
        angle = np.random.uniform(-5, 5)

        center = (
            image.shape[1] / 2,
            image.shape[0] / 2,
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0,
        )

        image = cv2.warpAffine(
            image,
            matrix,
            (image.shape[1], image.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    # Resize for ResNet-18.
    image = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )

    # Convert [0, 255] -> [0, 1].
    image = image.astype(np.float32) / 255.0

    # Add channel dimension.
    image = np.expand_dims(
        image,
        axis=0,
    )

    return image