
import cv2
import time
import csv
from pathlib import Path

from src.detection.face_detector_haar import detect_faces
from src.detection.face_detector_yunet import detect_faces_yunet

# Dossier contenant les images
DATASET_PATH = Path("data/raw/comparison/document_mobile")

# Fichier de résultats
OUTPUT_PATH = Path("experiments/detection_dataset_results.csv")

categories = ["normal", "inclined", "difficult"]

results = []

# Initialisation des détecteurs avant les mesures
print("Initialisation des détecteurs...")

for category in categories:
    folder = DATASET_PATH / category

    for image_path in sorted(folder.iterdir()):
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Image illisible : {image_path.name}")
            continue

        # Warm-up
        detect_faces(image)
        detect_faces_yunet(image, confidence=0.5)

print("\nDébut de la comparaison...\n")

for category in categories:
    folder = DATASET_PATH / category

    for image_path in sorted(folder.iterdir()):
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        # Test Haar Cascade
        start = time.perf_counter()
        haar_faces = detect_faces(image)
        haar_time = (time.perf_counter() - start) * 1000

        # Test YuNet
        start = time.perf_counter()
        yunet_faces = detect_faces_yunet(
            image,
            confidence=0.5
        )
        yunet_time = (time.perf_counter() - start) * 1000

        haar_count = len(haar_faces)
        yunet_count = len(yunet_faces)

        results.append({
            "image": image_path.name,
            "category": category,
            "haar_faces": haar_count,
            "yunet_faces": yunet_count,
            "yunet_confidence": 0.5,
            "haar_time_ms": round(haar_time, 2),
            "yunet_time_ms": round(yunet_time, 2)
        })

        print(f"Image : {image_path.name}")
        print(f"  Haar : {haar_count} visage(s) - {haar_time:.2f} ms")
        print(f"  YuNet : {yunet_count} visage(s) - {yunet_time:.2f} ms")

# Enregistrer les résultats
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

print("\n===== COMPARAISON TERMINÉE =====")
print(f"Nombre d'images testées : {len(results)}")
print(f"Résultats enregistrés dans : {OUTPUT_PATH}")
