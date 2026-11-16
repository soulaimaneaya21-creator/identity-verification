import cv2
import matplotlib.pyplot as plt

from src.utils.image_utils import load_image
from src.detection.face_detector_haar import (
    detect_faces,
    draw_faces
)

# 1. Charger l'image
image = load_image("data/raw/test_image1.png")

# 2. Détecter les visages
faces = detect_faces(image)

# 3. Afficher le nombre de visages
print("Nombre de visages détectés :", len(faces))

# 4. Afficher le résultat
result = draw_faces(image, faces)

# Conversion BGR vers RGB
result_rgb = cv2.cvtColor(
    result,
    cv2.COLOR_BGR2RGB
)

plt.imshow(result_rgb)
plt.title("Détection faciale avec Haar Cascade")
plt.axis("off")
plt.show()

# 5. Vérifier le résultat
if len(faces) == 0:
    print("Aucun visage détecté.")
else:
    print("Détection terminée.")