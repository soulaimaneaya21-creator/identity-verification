import cv2
import matplotlib.pyplot as plt

from src.utils.image_utils import load_image

from src.detection.face_detector_yunet import (
    detect_faces_yunet,
    draw_faces_yunet
)

# 1. Charger l'image
image = load_image("data/raw/test_image1.png")

# 2. Détecter les visages
faces = detect_faces_yunet(image)

# 3. Afficher le nombre de visages
print("Nombre de visages détectés avec YuNet :", len(faces))

# 4. Afficher les scores
for i, face in enumerate(faces):
    print(
        f"Visage {i + 1} - Confiance : {face[14]:.4f}"
    )

# 5. Dessiner les résultats
result = draw_faces_yunet(image, faces)

# 6. Convertir BGR en RGB
result_rgb = cv2.cvtColor(
    result,
    cv2.COLOR_BGR2RGB
)

# 7. Afficher l'image
plt.imshow(result_rgb)
plt.title("Détection faciale avec YuNet")
plt.axis("off")
plt.show()