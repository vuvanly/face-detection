import shutil

from facenet_pytorch import training
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import re

import helper
import settings
from custom_image_folder import CustomImageFolder


def get_best_train_val_models():
    train_model_path = settings.trained_model_path()

    train_models = [f for f in os.listdir(train_model_path)]
    train_models_with_accuracy = {}
    for model_file in train_models:
        match = re.search(r"\d+\.\d+", model_file)
        if match:
            accuracy = float(match.group())
            model_path = os.path.join(train_model_path, model_file)
            train_models_with_accuracy[model_path] = accuracy

    train_models_with_accuracy = dict(sorted(train_models_with_accuracy.items(), key=lambda x: x[1], reverse=True))
    best_train_models = list(train_models_with_accuracy.keys())[:5]

    val_model_path = settings.validated_model_path()

    val_models = [f for f in os.listdir(val_model_path)]
    val_models_with_accuracy = {}
    for model_file in val_models:
        match = re.search(r"\d+\.\d+", model_file)
        if match:
            accuracy = float(match.group())
            model_path = os.path.join(val_model_path, model_file)
            val_models_with_accuracy[model_path] = accuracy

    val_models_with_accuracy = dict(sorted(val_models_with_accuracy.items(), key=lambda x: x[1], reverse=True))
    best_val_models = list(val_models_with_accuracy.keys())[:5]
    return best_train_models + best_val_models


data_dir = settings.dataset_path()
test_dir = os.path.join(data_dir, "test")

batch_size = 32
epochs = 20
workers = 0  # if os.name == 'nt' else 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trans = helper.resnet_transform()
test_dataset = CustomImageFolder(test_dir, transform=trans)

test_loader = DataLoader(test_dataset, num_workers=workers, batch_size=batch_size)

loss_fn = torch.nn.CrossEntropyLoss()
metrics = {"fps": training.BatchTimer(), "acc": training.accuracy}

writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10
models_with_accuracy = {}
for model_file in get_best_train_val_models():
    resnet = torch.load(model_file)

    resnet.eval()

    test_loss, test_metrics = training.pass_epoch(
        resnet, loss_fn, test_loader, batch_metrics=metrics, show_running=True, device=device, writer=writer
    )
    models_with_accuracy[model_file] = test_metrics["acc"].item() * 100

models_with_accuracy = dict(sorted(models_with_accuracy.items(), key=lambda x: x[1], reverse=True))
best_model = list(models_with_accuracy.keys())[0]
shutil.copy(best_model, settings.model_path())
