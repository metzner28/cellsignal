# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader_gpu import CellSignalDataset
from torch.utils.data import DataLoader, random_split
import time
import copy
from tqdm.auto import tqdm
from squeezenet import get_squeezenet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

# %%
# modified squeezenet with 6 channels in the first layer 
# https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/squeezenet.py

model = get_squeezenet('1.1', in_channels = 6, in_size = (512,512))
model.cuda()

# %%
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, epochs = 25):

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
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
model_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)
dataset_sizes = {
    'train': len(train),
    'val': len(val)
}

# %%
params = {
        'model': model,
        'dataloaders': dataloaders,
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': model_lr_scheduler,
        'dataset_sizes': dataset_sizes,
        'epochs': 20
}

try: 
    model, history = train_model(**params)

finally:
    torch.save(model.state_dict(),'squeezenet_trained.pt')
    history = {k: v.cpu() for k,v in history.items()}
    df_model = pd.DataFrame(history)
    df_model.to_csv("model_squeezenet.csv", index = False)