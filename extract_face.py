import os

import settings
from image_processor import ImageProcessor


def extract_face(input_image_folder_path: str, output_image_folder_path: str):
    image_processor = ImageProcessor()
    image_files = [f for f in os.listdir(input_image_folder_path) if f.endswith(".jpg") or f.endswith(".png")]

    for image_file in image_files:
        current_image_path = os.path.join(input_image_folder_path, image_file)
        new_image_path = os.path.join(output_image_folder_path, image_file)
        image_processor.extract_face(current_image_path, new_image_path)


def resize(image_folder_path: str):
    image_processor = ImageProcessor()
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith(".jpg") or f.endswith(".png")]

    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)
        image_processor.resize_to_square_image(image_path, image_path, 320)


def main():
    os.makedirs(settings.cropped_images_path(), exist_ok=True)
    extract_face(settings.images_path(), settings.cropped_images_path())
    resize(settings.cropped_images_path())


if __name__ == "__main__":
    main()
