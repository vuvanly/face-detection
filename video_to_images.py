import cv2
import os

import helper
import settings


def convert_video_to_images(input_video_folder, output_image_folder):
    video_files = [f for f in os.listdir(input_video_folder) if f.endswith(".mp4")]

    for video_file in video_files:
        video_path = os.path.join(input_video_folder, video_file)
        person_id = helper.names_to_classes()[video_file.split(".mp4")[0]]

        # Create a video capture object
        cap = cv2.VideoCapture(video_path)

        # Check if video capture object is successfully created
        if not cap.isOpened():
            print("Error opening video file: {}".format(video_file))
            continue

        # Get the frame rate of the video
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Loop through all the frames in the video
        frame_count = 0
        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # Check if frame is successfully read
            if not ret:
                break

            # Create the filename for the image
            image_name = "{}_{}.jpg".format(person_id, frame_count)

            # Save the frame as an image
            image_path = os.path.join(output_image_folder, image_name)
            cv2.imwrite(image_path, frame)

            # Increment the frame count
            frame_count += 1

        # Release the video capture object
        cap.release()
        print("Converted {} into {} images.".format(video_file, frame_count))


def main():
    os.makedirs(settings.images_path(), exist_ok=True)
    convert_video_to_images(settings.video_path(), settings.images_path())


if __name__ == "__main__":
    main()
