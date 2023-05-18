from mtcnn import MTCNN
import cv2
import numpy as np
class ImageProcessor:
    def __init__(self):
        self.detector = MTCNN()
    def extract_face(self, input_image_filepath: str, output_image_filepath):
        # Load the image
        image = cv2.imread(input_image_filepath)

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self.detector.detect_faces(image)

        # Loop through all the faces found in the image
        for face in faces:
            # Get the coordinates of the face
            x1, y1, width, height = face['box']
            x2, y2 = x1 + width, y1 + height

            # Crop the face out of the image
            face_image = image[y1:y2, x1:x2]
            cv2.imwrite(output_image_filepath, face_image)

    @staticmethod
    def resize_to_square_image(input_image_filepath: str, output_image_filepath: str, desired_length: int):
        image = cv2.imread(input_image_filepath)
        if image is None:
            return

        height, width, _ = image.shape

        # Set the desired width of the image
        if height > width:
            desired_height = desired_length
            desired_width = int(desired_height * width / float(height))
            y_offset = 0
            x_offset = int((desired_length - desired_width) / 2)
        else:
            desired_width = desired_length
            desired_height = int(desired_width * height / float(width))
            x_offset = 0
            y_offset = int((desired_length - desired_height) / 2)
        desired_size = (desired_width, desired_height)
        resized_image = cv2.resize(image, desired_size)
        # fill with empty space
        blank_image = np.zeros((desired_length, desired_length, 3), np.uint8)
        blank_image[:, :] = (255, 255, 255)

        l_img = blank_image.copy()
        l_img[y_offset:y_offset + desired_height, x_offset:x_offset + desired_width] = resized_image.copy()
        cv2.imwrite(output_image_filepath, l_img)