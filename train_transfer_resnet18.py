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

def train_model(model, data_loaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    writer = SummaryWriter('runs/pretrained_resnet18')

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in    range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
                print('Training: ')
                phase_acc_name = 'trainAcc'
                phase_loss_name = 'trainLoss'
            else:
                print('Validation: ')
                model.eval()   # Set model to evaluate mode
                phase_acc_name = 'valAcc'
                phase_loss_name = 'valLoss'

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # ind = 0
            for inputs, labels in tqdm(data_loaders[phase]):
                # ind +=1 

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # if ind > 3:
                #     break  

            epoch_loss = running_loss / len(data_loaders[phase])
            epoch_acc = running_corrects.double() / len(data_loaders[phase])
            
            # Tensorboard
            writer.add_scalar(phase_acc_name,epoch_acc, epoch)

            # writer.add_scalars(phase_loss_name, epoch_loss, epoch) 

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, 'weights/pretrained_resnet18.pth')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model

# Load data
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

model = torchvision.models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.Adam(model.parameters(), lr=0.001)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

data_loaders = {'train': train_loader, 'val': val_loader}
model = train_model(model, data_loaders, criterion, optimizer_conv, exp_lr_scheduler,
                       num_epochs=20) 