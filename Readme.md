## Analysis
Raw training data: list video of target people
Expected input: image of a person
Expected output: print the name of recognized person

First we need to convert video to images to prepare training data.
There are multiple ways to approach:
* Use CNN to build our own model with prepared images
* Extract face from prepared images, use these new data with CNN to build our own model
* Use pretrained model with prepared images
* Extract face from prepared images, use these new data with pretrained model



### Face Extraction Libraries
* dlib
* mtcnn (chosen for this project)
* retinaface

### Pretrained Models
* https://github.com/deepinsight/insightface
* https://github.com/serengil/deepface
* https://github.com/davidsandberg/facenet (chosen for this project)
* https://github.com/ZhaoJ9014/face.evoLVe


## Train steps for first model

### Prepare the `Video` folder and copy the videos of target people into it.

### Get meta data from videos
Run script `video_to_metadata.py`. It will generate 2 files:
* meta_name_to_id.json: File contains mapping from name to id (or class of model)
* meta_id_to_name.json: File contains mapping from id (or class of model) to name

### Prepare `class_indexes.json` file
This file will be used during training to map from the model's class to the predicted index.

### Convert video to images
Run script `video_to_images.py` to extract image frames from videos

### Extract face from images
Run script `extract_face.py` to extract face from image into new image

### Split images into train, validation, test folder
Run script `split_images_to_dataset.py` to split into train, validation, test subset with rate: 60:20:20

### Finetune with model facenet
Run script `finetune_model.py` to finetune by pretrained model with our own dataset 
https://github.com/davidsandberg/facenet

### Test to get best model
Run script `test_model.py` to test the best 5 trained models, and the best 5 validated models, to get the best performance model
This model will be used for detection

## Face Detection
Run script `face_detect.py image_path|folder_path` to detect face from an image, or images from a folder

## How to add new class
* Copy video of new person to folder `NextVideo`
* Update `meta_name_to_id.json`, `meta_id_to_name.json`, `class_indexes.json` file to add new class
* Run script `prepare_additional_data.py` to update data of new class to current dataset
* Run script `update_model.py` to update model with new dataset of new class
* Run script `test_model.py` to test new trained and validated model to get best model

## Next improvement
* Use GAN or diffusion to generate more training dataset