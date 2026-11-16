
import cv2
from pathlib import Path

MODEL_PATH = Path(
    "models/face_detection_yunet_2023mar_int8.onnx"
)

# Cache des détecteurs YuNet initialisés
detectors = {}


def detect_faces_yunet(image,  confidence=0.5):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle YuNet introuvable : {MODEL_PATH}"
        )

    height, width = image.shape[:2]

    # Utiliser les dimensions et le seuil comme clé
    key = (width, height, confidence)

    # Créer le détecteur seulement s'il n'existe pas déjà
    if key not in detectors:

        detectors[key] = cv2.FaceDetectorYN.create(
            str(MODEL_PATH),
            "",
            (width, height),
            confidence,
            0.3,
            5000
        )

    detector = detectors[key]

    # Effectuer la détection
    _, faces = detector.detect(image)

    if faces is None:
        return []

    return faces


def draw_faces_yunet(image, faces):

    result = image.copy()

    for face in faces:

        x, y, w, h = face[:4].astype(int)

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        landmarks = face[4:14].reshape(5, 2)

        for point in landmarks:
            px, py = point.astype(int)

            cv2.circle(
                result,
                (px, py),
                3,
                (0, 0, 255),
                -1
            )

        score = face[14]

        cv2.putText(
            result,
            f"{score:.2f}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return result
