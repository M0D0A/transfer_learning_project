import torch
import os
import numpy as np
import torch.nn as nn
from torchvision import models


def get_last_model(path):
    file_list = os.listdir(path)
    idx = np.array([int(num.split("_")[-1].split(".")[0]) for num in file_list]).argmax()
    return os.path.join(path,file_list[idx])


def create_model(num_classes):
    resnet50 = models.resnet50()
    fc = nn.Sequential(
        nn.Linear(in_features=2048,out_features=512, bias=False),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=512,out_features=128, bias=False),
        nn.BatchNorm1d(128),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=128, out_features=num_classes)
    )
    resnet50.fc = fc
    return resnet50


def create_tl_model(num_classes):
    model = create_model(num_classes)
    pretrained_weights = torch.load("models/pretrained/resnet50-0676ba61.pth", map_location="cpu")
    filtered_weights = {k: v for k, v in pretrained_weights.items() if k in model.state_dict()}
    # Параметр strick=False пропускает неслвпадающие ключи возвращает (списки пропущенных и лишних ключей)
    model.load_state_dict(filtered_weights, strict=False)

    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def create_ft_model(num_classes):# Доделать после обозначения путей и формата сохранения
    model = create_model(num_classes)
    path = get_last_model("models/save_tl_models")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])

    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.fc.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

    # for param in model.layer3.parameters():
    #     param.requires_grad = True

    return model
