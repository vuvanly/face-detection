import json
import os

import settings


def get_and_save_meta_data(video_folder):
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

    meta_id_to_name = {}
    meta_name_to_id = {}
    for index, video_file in enumerate(video_files):
        name = os.path.splitext(os.path.basename(video_file))[0]
        id = index + 1
        meta_name_to_id[name] = id
        meta_id_to_name[id] = name

    with open("meta_name_to_id.json", "w", encoding="utf-8") as json_file:
        json.dump(meta_name_to_id, json_file, ensure_ascii=False, indent=4)
    with open("meta_id_to_name.json", "w", encoding="utf-8") as json_file:
        json.dump(meta_id_to_name, json_file, ensure_ascii=False, indent=4)


def main():
    video_folder = settings.video_path()
    get_and_save_meta_data(video_folder)


if __name__ == "__main__":
    main()
