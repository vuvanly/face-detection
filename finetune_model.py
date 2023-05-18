from facenet_pytorch import InceptionResnetV1, training
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import os

import helper
import settings
from custom_image_folder import CustomImageFolder


def train_model(resnet, train_dataset, val_dataset, epochs):
    batch_size = 32
    workers = 0  # if os.name == 'nt' else 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, [5, 10])

    train_loader = DataLoader(train_dataset, num_workers=workers, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=workers, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {"fps": training.BatchTimer(), "acc": training.accuracy}

    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    print("\n\nInitial")
    print("-" * 10)
    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader, batch_metrics=metrics, show_running=True, device=device, writer=writer
    )

    for epoch in range(epochs):
        print("\nEpoch {}/{}".format(epoch + 1, epochs))
        print("-" * 10)

        resnet.train()
        train_loss, train_metrics = training.pass_epoch(
            resnet,
            loss_fn,
            train_loader,
            optimizer,
            scheduler,
            batch_metrics=metrics,
            show_running=True,
            device=device,
            writer=writer,
        )

        resnet.eval()

        val_loss, val_metrics = training.pass_epoch(
            resnet, loss_fn, val_loader, batch_metrics=metrics, show_running=True, device=device, writer=writer
        )

        train_model_local_path = "{0}/resnet_epoch_{1}_train_{2}".format(
            settings.trained_model_path(), epoch, train_metrics["acc"].item() * 100
        )
        val_model_local_path = "{0}/resnet_epoch_{1}_eval_{2}".format(
            settings.validated_model_path(), epoch, val_metrics["acc"].item() * 100
        )
        torch.save(resnet, train_model_local_path)
        torch.save(resnet, val_model_local_path)

    writer.close()


def main():
    data_dir = settings.dataset_path()
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 8

    trans = helper.resnet_transform()
    train_dataset = CustomImageFolder(train_dir, transform=trans)
    val_dataset = CustomImageFolder(val_dir, transform=trans)

    resnet = InceptionResnetV1(classify=True, pretrained="vggface2", num_classes=len(train_dataset.class_to_idx)).to(
        device
    )
    train_model(resnet, train_dataset, val_dataset, epochs)


if __name__ == "__main__":
    main()
