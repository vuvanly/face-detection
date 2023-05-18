import helper


def settings() -> dict:
    return helper.load_json_file("settings.json")


def video_path() -> str:
    return settings()["video_path"]


def images_path() -> str:
    return settings()["images_path"]


def cropped_images_path() -> str:
    return settings()["cropped_images_path"]


def dataset_path() -> str:
    return settings()["dataset_path"]


def trained_model_path() -> str:
    return settings()["trained_model_path"]


def validated_model_path() -> str:
    return settings()["validated_model_path"]


def model_path() -> str:
    return settings()["model_path"]
