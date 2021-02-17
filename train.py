#!/usr/bin/env python
# coding: utf-8
from PIL import Image
import cv2
from path import Path
import collections
import torch
import torch.backends.cudnn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
import segmentation_models_pytorch as smp
from utils.datasets import SlippyMapTilesConcatenation
from utils.loss import CrossEntropyLoss2d, mIoULoss2d, FocalLoss2d, LovaszLoss2d
from utils.transforms import (
    JointCompose,
    JointTransform,
    JointRandomHorizontalFlip,
    JointRandomRotation,
    ConvertImageMode,
    ImageToTensor,
    MaskToTensor,
)
from torchvision.transforms import Resize, CenterCrop, Normalize
from utils.metrics import Metrics
from models.segnet.segnet import segnet
from models.unet.unet import UNet
from torch import nn
import random
import os
import tqdm
import json
import pathlib
device = 'cuda'
path = ''

def get_model(num_classes):
#   model = UNet( num_classes = num_classes )
#   model = segnet(  n_classes = num_classes )
    model =smp.PSPNet(classes= num_classes )
    model.train()
    return model.to(device)

num_classes = 2
net = get_model(num_classes)
history = collections.defaultdict(list)
learning_rate = 5e-3
num_epochs = 50
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# criterion = mIoULoss2d().to(device)
criterion = nn.CrossEntropyLoss()


#net = torch.load('model/0514pspnet.pth')
def get_dataset_loaders( workers):
    target_size = 512
    batch_size = 4  

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform = JointCompose(
        [
            JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            JointTransform(Resize(target_size, Image.BILINEAR), Resize(target_size, Image.NEAREST)),
            JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
            JointRandomHorizontalFlip(0.5),
            JointRandomRotation(0.5, 90),
            JointTransform(ImageToTensor(), ImageToTensor()),
            JointTransform(Normalize(mean=mean, std=std), None),
        ]
    )
    train_dataset = SlippyMapTilesConcatenation(
        os.path.join(path, "training", "images"), os.path.join(path, "training", "labels"), transform,debug = False
    )

    val_dataset = SlippyMapTilesConcatenation(
        os.path.join(path, "validation", "images"), os.path.join(path, "validation", "labels"), transform,debug = False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=workers)

    return train_loader, val_loader



def train(loader, num_classes, device, net, optimizer, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    net.train()
    for images, masks  in tqdm.tqdm(loader):
        images = images.to(device)
        masks = masks.to(device).long()
        # assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

        num_samples += int(images.size(0))

        optimizer.zero_grad()
        outputs = net(images)

        # assert outputs.size()[2:] == masks.size()[1:], "resolutions or predictions and masks are in sync"
        # assert outputs.size()[1] == num_classes, "classes for predictions and dataset are in sync"
        loss = criterion(outputs, masks)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            prediction = output.detach()
            metrics.add(mask, prediction)

    # assert num_samples > 0, "dataset contains training images and labels"

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
    }

def validate(loader, num_classes, device, net, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    net.eval()

    for images, masks,  in tqdm.tqdm(loader):
        images = images.to(device)
        masks = masks.to(device)

        assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

        num_samples += int(images.size(0))

        outputs = net(images)

        assert outputs.size()[2:] == masks.size()[1:], "resolutions for predictions and masks are in sync"
        assert outputs.size()[1] == num_classes, "classes for predictions and dataset are in sync"

        loss = criterion(outputs, masks)

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            metrics.add(mask, output)

    assert num_samples > 0, "dataset contains validation images and labels"

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
    }

         
def main():

    train_loader, val_loader = get_dataset_loaders( 5)
    print(len(train_loader))
    for epoch in range(num_epochs):
        print("Epoch: {}/{}".format(epoch + 1, num_epochs))

        train_hist = train(train_loader, num_classes, device, net, optimizer, criterion)
        print( 'loss',train_hist["loss"],
                'miou',train_hist["miou"],
                'fg_iou',train_hist["fg_iou"],
                'mcc',train_hist["mcc"] )

        for k, v in train_hist.items():
            history["train " + k].append(v)

        val_hist = validate(val_loader, num_classes, device, net, criterion)
        print('loss',val_hist["loss"],
                'miou',val_hist["miou"],
                'fg_iou',val_hist["fg_iou"],
                'mcc',val_hist["mcc"])

        for k, v in val_hist.items():
            history["val " + k].append(v)

        pathlib.Path(os.path.join('model' )).mkdir(parents=True, exist_ok=True) 

        torch.save(net,f'model/psp_kinds_{num_epochs}_epoch.pth')

    json = json.dumps(history)
    f = open("model/psp_kindsest0607.json","w")
    f.write(json)
    f.close()



if __name__ == '__main__':
    main()