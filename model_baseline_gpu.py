# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import os
from data_loader_gpu import CellSignalDataset
from torch.utils.data import DataLoader, random_split
import time
import copy
from tqdm.auto import tqdm

# %%
# md_train = pd.read_csv("md_train.csv")
# md_downsampled = md_train[md_train["site"] == 1]
# df_downsampled["path"] =  df_downsampled.apply(lambda df: df["path"].replace("/Volumes/DRIVE/rxrx1/", "/home/ubuntu/cellsignal/"), axis = 1)
# df_downsampled.to_csv("df_downsampled.csv", index = False)

# %%
data = CellSignalDataset('md_downsampled.csv', transform = None)
n_classes = len(np.unique(data.img_labels["sirna_id"]))
TRAIN_VAL_SPLIT = 0.8
n_train = round(len(data) * TRAIN_VAL_SPLIT)
n_val = round(len(data) * (1 - TRAIN_VAL_SPLIT))
assert n_train + n_val == len(data)

train, val = random_split(data, lengths = [n_train, n_val], generator = torch.Generator().manual_seed(42))
train_loader = DataLoader(train, batch_size = 32, shuffle = True)
val_loader = DataLoader(val, batch_size = 32, shuffle = True)

dataset_sizes = {'train': len(train), 'val': len(val)}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_classes = len(np.unique(data.img_labels["sirna_id"]))

# %%
# https://stackoverflow.com/questions/62629114/how-to-modify-resnet-50-with-4-channels-as-input-using-pre-trained-weights-in-py

# pretrained weights option
model = torchvision.models.resnet18()
# for param in model.parameters():
#      param.requires_grad = False

# modify input layer to accept 6 channels
trained_kernel = model.conv1.weight
new_conv = nn.Conv2d(6, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
with torch.no_grad():
    new_conv.weight[:] = torch.stack([torch.mean(trained_kernel, 1)] * 6, dim = 1)
model.conv1 = new_conv

# modify output layer so probabilities for each sirna class are predicted
model.fc = nn.Linear(model.fc.in_features, n_classes)

model = model.to(device)

# %%
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, epochs = 20):

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    since = time.time()
    best_model_wts= copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, _, labels in tqdm(dataloaders[phase]):
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            elif phase == 'val':
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# %%
dataloaders = {'train': train_loader, 'val': val_loader}
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
model_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)

# %%
params = {
    'model': model,
    'dataloaders': dataloaders,
    'criterion': criterion,
    'optimizer': optimizer,
    'scheduler': model_lr_scheduler,
    'dataset_sizes': dataset_sizes
}

try:
    model, history = train_model(**params)

finally:
    torch.save(model.state_dict(), 'resnet18_baseline_fully_trained.pt')
    history = {k: v.cpu() for k,v in history.items()}
    df_model = pd.DataFrame(history)
    df_model.to_csv("resnet18_baseline_fully_trained.csv", index = False)
