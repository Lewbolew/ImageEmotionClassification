import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataloaders.standart_dataloader import EmotionImagesDataset
from utils.training_functions import *

# # Load data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

emotions_train = EmotionImagesDataset('/home/petryshak/ImageEmotionClassification/dataset/b-t4sa_train.txt', 
                                       '/home/petryshak/ImageEmotionClassification/dataset/')
train_loader = torch.utils.data.DataLoader(emotions_train,
                                             batch_size=128, 
                                             shuffle=True,
                                            )

emotions_val = EmotionImagesDataset('/home/petryshak/ImageEmotionClassification/dataset/b-t4sa_val.txt', 
                                       '/home/petryshak/ImageEmotionClassification/dataset/')
val_loader = torch.utils.data.DataLoader(emotions_val,
                                             batch_size=128, 
                                             shuffle=False,
                                            )
# Define the model


model = torchvision.models.vgg19_bn(pretrained=False)


for param in model.features.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(4096,3)

model.load_state_dict(torch.load('weights/pretrained_vgg19_bn.pth'))

model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.Adam(model.parameters(), lr=0.001)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

data_loaders = {'train': train_loader, 'val': val_loader}
model = train_model(model, data_loaders, criterion, optimizer_conv, exp_lr_scheduler,
                       'weights/pretrained_vgg19_bn.pth', device, num_epochs=30) 