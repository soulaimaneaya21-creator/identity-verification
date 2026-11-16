
import cv2
from pathlib import Path

from src.detection.face_detector_haar import (
    detect_faces,
    draw_faces
)

from src.detection.face_detector_yunet import (
    detect_faces_yunet,
    draw_faces_yunet
)

DATASET_PATH = Path("data/raw/comparison/document_mobile")
OUTPUT_PATH = Path("experiments/visual_comparison")

categories = ["normal", "inclined", "difficult"]


def resize_image(image, max_width=900):
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    ratio = max_width / width
    new_size = (max_width, int(height * ratio))

    return cv2.resize(image, new_size)


OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

for category in categories:

    folder = DATASET_PATH / category

    for image_path in sorted(folder.iterdir()):

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        # Détection
        haar_faces = detect_faces(image)
        yunet_faces = detect_faces_yunet(
            image,
            confidence=0.5
        )

        # Dessiner les rectangles
        haar_result = draw_faces(image, haar_faces)
        yunet_result = draw_faces_yunet(image, yunet_faces)

        # Redimensionner pour faciliter la visualisation
        haar_result = resize_image(haar_result)
        yunet_result = resize_image(yunet_result)

        # Ajouter les titres
        cv2.putText(
            haar_result,
            "HAAR CASCADE",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

        cv2.putText(
            yunet_result,
            "YUNET",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

        # Comparaison côte à côte
        comparison = cv2.hconcat([
            haar_result,
            yunet_result
        ])

        output_file = OUTPUT_PATH / f"{image_path.stem}_comparison.jpg"

        cv2.imwrite(str(output_file), comparison)

        print("Image enregistrée :", output_file)

print("\nComparaison visuelle terminée.")
