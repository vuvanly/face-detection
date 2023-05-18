import os
import shutil

import settings
import video_to_images
from extract_face import extract_face, resize
from split_images_to_dataset import split_dataset


def move_files(from_folder: str, to_folder: str, file_pattern: str):
    # Get a list of files matching the pattern in the source folder
    files = [file for file in os.listdir(from_folder) if file.endswith(file_pattern)]

    # Move each file to the destination folder
    for file in files:
        source_path = os.path.join(from_folder, file)
        destination_path = os.path.join(to_folder, file)
        shutil.move(source_path, destination_path)


def move_files_keep_subdir(from_folder: str, to_folder: str, file_pattern: str):
    # Iterate through the source folder and its subdirectories
    for root, dirs, files in os.walk(from_folder):
        for file in files:
            if file.endswith(file_pattern):
                source_path = os.path.join(root, file)
                relative_path = os.path.relpath(source_path, from_folder)
                destination_path = os.path.join(to_folder, relative_path)
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                shutil.move(source_path, destination_path)


video_folder = "NextVideo"
image_folder = "NextImage"
cropped_image_folder = "CroppedNextImage"
dataset = "next_dataset"
os.makedirs(image_folder, exist_ok=True)
video_to_images.convert_video_to_images(video_folder, image_folder)
os.makedirs(cropped_image_folder, exist_ok=True)
extract_face(image_folder, cropped_image_folder)
resize(cropped_image_folder)
os.makedirs(dataset, exist_ok=True)
split_dataset(cropped_image_folder, dataset)

# Merge next data to current data
move_files(video_folder, settings.video_path(), ".mp4")
move_files(image_folder, settings.images_path(), ".jpg")
move_files(cropped_image_folder, settings.cropped_images_path(), ".jpg")
move_files_keep_subdir(dataset, settings.dataset_path(), ".jpg")
