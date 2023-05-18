import glob
import os
import shutil

import helper
import settings


def split_dataset(images_path: str, dataset_path: str):
    image_folder = images_path
    train_folder = os.path.join(dataset_path, "train")
    val_folder = os.path.join(dataset_path, "val")
    test_folder = os.path.join(dataset_path, "test")

    for i in list(helper.names_to_classes().values()):
        for folder in [train_folder, val_folder, test_folder]:
            class_folder = os.path.join(folder, str(i))
            os.makedirs(class_folder, exist_ok=True)

        regex_pattern = "{}_*.jpg".format(i)
        image_files = glob.glob(os.path.join(image_folder, regex_pattern))
        image_files = sorted(image_files, key=lambda x: (len(x), x))
        length = len(image_files)
        # split into train:val:test with rate: 60:20:20
        start_val_index = int(length * 0.6)
        start_test_index = int(length * 0.8)
        for index, image_file in enumerate(image_files):
            if index >= start_test_index:
                dest_path = os.path.join(val_folder, str(i))
            elif index >= start_val_index:
                dest_path = os.path.join(test_folder, str(i))
            else:
                dest_path = os.path.join(train_folder, str(i))
            shutil.copy(image_file, dest_path)


def main():
    split_dataset(settings.cropped_images_path(), settings.dataset_path())


if __name__ == "__main__":
    main()
