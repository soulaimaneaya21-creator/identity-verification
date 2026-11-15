import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Définir le chemin de l'image
image_path = Path("data/raw/test_image.png"
"")

# 2. Vérifier que le fichier existe
if not image_path.exists():
    print("Erreur : l'image n'existe pas.")
else:
    # 3. Lire l'image avec OpenCV
    image = cv2.imread(str(image_path))

    # 4. Vérifier que l'image est correctement chargée
    if image is None:
        print("Erreur : impossible de lire cette image.")
    else:
        # 5. Récupérer les dimensions
        hauteur, largeur, canaux = image.shape

        print("Image chargée avec succès !")
        print("Largeur :", largeur, "pixels")
        print("Hauteur :", hauteur, "pixels")
        print("Nombre de canaux :", canaux)

        # 6. Convertir BGR vers RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 7. Afficher l'image
        plt.imshow(image_rgb)
        plt.title("Image de test")
        plt.axis("off")
        plt.show()