
import cv2

# 1. Charger le détecteur une seule fois
cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

# Vérifier le chargement
if face_cascade.empty():
    raise RuntimeError("Impossible de charger le détecteur Haar.")


def detect_faces(image):

    # 2. Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Détecter les visages
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    return faces


def draw_faces(image, faces):

    result = image.copy()

    for (x, y, width, height) in faces:
        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

    return result
