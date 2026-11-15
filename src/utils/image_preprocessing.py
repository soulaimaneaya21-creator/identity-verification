import cv2


def resize_image(image, width=640):
    height, original_width = image.shape[:2]

    if original_width <= width:
        return image

    ratio = width / original_width
    new_height = int(height * ratio)

    resized = cv2.resize(
        image,
        (width, new_height)
    )

    return resized


def convert_to_grayscale(image):
    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )