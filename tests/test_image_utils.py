from src.utils.image_utils import load_image, get_image_info

image = load_image("data/raw/test_image.png")

info = get_image_info(image)

print("Informations de l'image :")
print(info)