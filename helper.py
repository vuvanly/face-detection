import json
import os

import torchvision.transforms as transforms
from facenet_pytorch.models.mtcnn import fixed_image_standardization
import numpy as np


def classes_to_idx() -> dict:
    return load_json_file("class_indexes.json")


def classes_to_name() -> dict:
    return load_json_file("meta_id_to_name.json")


def names_to_classes() -> dict:
    return load_json_file("meta_name_to_id.json")


def idx_to_name(idx: int) -> str:
    for class_value, idx_value in classes_to_idx().items():
        if idx_value == idx:
            return classes_to_name()[class_value]

    return "Not Found"


def load_json_file(filepath: str):
    with open(filepath, "r") as json_file:
        return json.load(json_file)


def delete_all_files(folder_path: str):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Iterate through the files and delete them
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def resnet_transform():
    return transforms.Compose(
        [transforms.Resize((160, 160)), np.float32, transforms.ToTensor(), fixed_image_standardization]
    )
