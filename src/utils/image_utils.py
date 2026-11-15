import cv2
from pathlib import Path


def load_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image introuvable : {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            "Impossible de lire le fichier image."
        )

    return image


def get_image_info(image):
    height, width, channels = image.shape

    return {
        "width": width,
        "height": height,
        "channels": channels
    }