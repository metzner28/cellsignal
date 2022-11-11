# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.nn import functional as F
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
data = CellSignalDataset('../md_final.csv', transform = None)
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
n_cell_types = 4
# %%
# this creates the model without the classification layer
model = torchvision.models.resnet18()

trained_kernel = model.conv1.weight
new_conv = nn.Conv2d(6, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
with torch.no_grad():
    new_conv.weight[:] = torch.stack([torch.mean(trained_kernel, 1)] * 6, dim = 1)
model.conv1 = new_conv
# model.fc = nn.Linear(model.fc.in_features, n_classes)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
summary(model, input_size = (6,512,512))
# %%
class CellTypeModel(nn.Module):
    
    def __init__(self, inputs, cell_types, emb_dim = 512, n_classes = 1139):
        
        super().__init__()
        self.embedding = nn.Embedding(4, emb_dim)

        model = torchvision.models.resnet18()
        trained_kernel = model.conv1.weight
        new_conv = nn.Conv2d(6, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        with torch.no_grad():
            new_conv.weight[:] = torch.stack([torch.mean(trained_kernel, 1)] * 6, dim = 1)
        model.conv1 = new_conv
        self.features = nn.ModuleList(model.children())[:-1]
        self.features = nn.Sequential(*self.features)
        self.fc = nn.Linear(emb_dim, n_classes)

    
    def forward(self, x, cell_types):

        '''
        taking cell type as additional arg, embedding into vector with same dimensions as the output of the ResNet18 head, elementwise multiplying cell type embedding with head, taking ReLU of embedding * image vector, then classifying
        '''
        
        x = self.features(x)
        emb = self.embedding(cell_types)
        emb_x = x * emb
        output = F.relu(emb_x)
        output = self.fc(output)

        return output
