import torch.nn as nn

import torch
import os

import finetune_model
import helper
import settings
from custom_image_folder import CustomImageFolder

# Load the pre-trained model
resnet = torch.load(settings.model_path())
# Freeze the weights of the existing layers
for param in resnet.parameters():
    param.requires_grad = False

num_classes = len(helper.classes_to_idx().values())

# Add a new output layer for the new label
new_output_layer = nn.Linear(resnet.logits.in_features, num_classes)

resnet.logits = new_output_layer

# Set the new output layer to trainable
for param in resnet.logits.parameters():
    param.requires_grad = True

data_dir = settings.dataset_path()
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

batch_size = 32
epochs = 4
workers = 0  # if os.name == 'nt' else 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trans = helper.resnet_transform()
train_dataset = CustomImageFolder(train_dir, transform=trans)
val_dataset = CustomImageFolder(val_dir, transform=trans)

helper.delete_all_files(settings.trained_model_path())
helper.delete_all_files(settings.validated_model_path())

finetune_model.train_model(resnet, train_dataset, val_dataset, epochs)
