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
from data_loader_normalized import CellSignalDataset
from torch.utils.data import DataLoader, random_split
import time
import copy
from tqdm.auto import tqdm

# %%
data = CellSignalDataset('../md_final.csv', transform = None, normalize_file = "../metadata_pixel_stats.csv")
n_classes = len(np.unique(data.img_labels["sirna_id"]))
TRAIN_VAL_SPLIT = 0.8
n_train = round(len(data) * TRAIN_VAL_SPLIT)
n_val = round(len(data) * (1 - TRAIN_VAL_SPLIT))
assert n_train + n_val == len(data)

train, val = random_split(data, lengths = [n_train, n_val], generator = torch.Generator().manual_seed(4242))
train_loader = DataLoader(train, batch_size = 32, shuffle = True)
val_loader = DataLoader(val, batch_size = 32, shuffle = True)

dataset_sizes = {'train': len(train), 'val': len(val)}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_classes = len(np.unique(data.img_labels["sirna_id"]))

# %%
class CellTypeModel(nn.Module):
    
    def __init__(self, emb_dim = 512, n_classes = 1139):
        
        super().__init__()

        # 4 cell types
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
        
        x = self.features(x).flatten(1)
        emb = self.embedding(cell_types)
        emb_x = x * emb
        output = F.relu(emb_x)
        output = self.fc(output)

        return output

# %%
model = CellTypeModel()
model.to(device)

# %%
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, epochs = 20):

    # 4 will throw an error once passed to embedding, which is the way it should be
    get_celltype = lambda ct: 0 if 'HUVEC' in ct else 1 if 'U2OS' in ct else 2 if 'HEPG2' in ct else 3 if 'RPE' in ct else 4
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
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

            for inputs, exp_labels, labels in tqdm(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)
                cell_types = [get_celltype(i) for i in exp_labels]
                cell_types = torch.LongTensor(cell_types).to(device)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, cell_types)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'val':
                scheduler.step(epoch_loss)

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(float(epoch_acc.detach().cpu()))
            elif phase == 'val':
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(float(epoch_acc.detach().cpu()))

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history


# %%
dataloaders = {'train': train_loader, 'val': val_loader}
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 0.0005)
model_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.2, patience = 2)

# %%
params = {
    'model': model,
    'dataloaders': dataloaders,
    'criterion': criterion,
    'optimizer': optimizer,
    'scheduler': model_lr_scheduler,
    'dataset_sizes': dataset_sizes
}

model, history = train_model(**params, epochs = 32)
df_model = pd.DataFrame(history)
df_model.to_csv("celltype_norm/cell_type_norm_ce.csv", index = False)

torch.save(model.state_dict(), 'celltype_norm/cell_type_norm_ce.pt')
