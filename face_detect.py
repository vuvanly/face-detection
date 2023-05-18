import sys

import torch
from PIL import Image
import os

import helper
import settings


def transform_image(image_path: str):
    image = Image.open(image_path)
    image = image.convert("RGB")
    transform = helper.resnet_transform()
    tensor_image = transform(image)
    # Add an extra dimension for the batch size
    tensor_image = tensor_image.unsqueeze(0)
    return tensor_image


def detect(image_path):
    model = torch.load(settings.model_path())
    model.classify = True
    tensor_image = transform_image(image_path)
    y_pred = model(tensor_image)
    predicted_class = torch.argmax(y_pred).item()
    return helper.idx_to_name(predicted_class)


def main():
    args = sys.argv
    if len(args) != 2:
        print("Invalid arguments. This command should have 1 argument is image_file_path or image_folder_path")
        exit(1)
    path = args[1]
    if os.path.isfile(path):
        output = detect(path)
        print(f"{path}: {output}")
    elif os.path.isdir(path):
        image_files = [f for f in os.listdir(path) if f.endswith(".jpg") or f.endswith(".png")]
        for image_file in image_files:
            image_path = os.path.join(path, image_file)
            output = detect(image_path)
            print(f"{image_path}: {output}")
    else:
        print("Invalid path. Exit!")
        exit(1)


if __name__ == "__main__":
    main()
