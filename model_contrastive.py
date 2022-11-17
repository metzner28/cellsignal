# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from data_loader_gpu import CellSignalDataset
from torch.utils.data import DataLoader, random_split
import time
import copy
from tqdm.auto import tqdm
from helpers_contrastive import *

# %%
md_path = '../md_final.csv'

data = CellSignalDataset(md_path, transform = None)
data_for_encoder = CellSignalDataset(md_path, transform = transforms.AugMix())
n_classes = len(np.unique(data.img_labels["sirna_id"]))
TRAIN_VAL_SPLIT = 0.8
n_train = round(len(data) * TRAIN_VAL_SPLIT)
n_val = round(len(data) * (1 - TRAIN_VAL_SPLIT))
assert n_train + n_val == len(data)

enc_loader = DataLoader(data_for_encoder, batch_size = 128, shuffle = True)

train, val = random_split(data, lengths = [n_train, n_val], generator = torch.Generator().manual_seed(42))
train_loader = DataLoader(train, batch_size = 32, shuffle = True)
val_loader = DataLoader(val, batch_size = 32, shuffle = True)

dataset_sizes = {'train': len(train), 'val': len(val), 'full': len(data)}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_classes = len(np.unique(data.img_labels["sirna_id"]))

# %%
model_encoder = ContrastiveCellTypeEncoder()
model_classifier = ContrastiveCellTypeClassifier()
model_encoder.to(device)
model_classifier.to(device)

# %%

def train_encoder(model, criterion, optimizer, dataloader, dataset_sizes, epochs = 50, scheduler = None, enc_state_dict = '../results/contrastive_encoder.pt'):

    # 4 will throw an error once passed to embedding, which is the way it should be
    get_celltype = lambda ct: 0 if 'HUVEC' in ct else 1 if 'U2OS' in ct else 2 if 'HEPG2' in ct else 3 if 'RPE' in ct else 4
    
    history = {
        'train_loss': [],
    }

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000
    
    for epoch in range(epochs):

        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0

        for loc, inputs, exp_labels, labels in tqdm(dataloader):

            # concatenate the transformed and original images from each batch into a multiview batch, per paper - ensures >1 positive per batch

            # the original and transformed images need to be next to each other for the loss fn to work - this is not how the dataloader returns them
            # fix that with python list magic here
            # the dataloader returns a 2-length list of tuples where the things we want next to each other are at the same position in each of the tuples

            # loc = [loc[n][i] for i in range(len(loc[0])) for n in range(len(loc))]
            inputs = [inputs[n][i] for i in range(len(inputs[0])) for n in range(len(inputs))]
            exp_labels = [exp_labels[n][i] for i in range(len(exp_labels[0])) for n in range(len(exp_labels))]

            inputs = torch.stack(inputs)


            # now do the training

            inputs = inputs.to(device)
            labels = labels.to(device)
            cell_types = [get_celltype(i) for i in exp_labels]
            cell_types = torch.LongTensor(cell_types).to(device)
            
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                
                outputs, _ = model(inputs, cell_types)
                
                # assuming output.shape is (64,128) and only 2 views per image
                # as long as the view outputs are next to each other within a batch this is right

                outputs = torch.reshape(outputs, (-1, 2, outputs.shape[-1]))
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / dataset_sizes['full']
            if scheduler:
                scheduler.step(epoch_loss)
          
            history['train_loss'].append(epoch_loss)
            print(f'Train loss: {epoch_loss:.4f}')

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best loss: {best_loss:4f}')

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), enc_state_dict)
    
    return history

# %%    
def train_classifier(encoder, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, epochs = 25, enc_state_dict = '../results/contrastive_encoder.pt', save_path = '../results/contrastive_classifier.pt'):

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

    try:
        encoder.load_state_dict(
            torch.load(enc_state_dict)
        ) 
        encoder.eval()  
    except:
        raise FileNotFoundError("encoder parameters don't exist at specified path")

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

                with torch.no_grad():
                    _, inputs = encoder(inputs)

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
                scheduler.step()

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
    torch.save(model.state_dict(), save_path)
    return history

# %%
contrastive = SupConLoss()
lars = LARS(model_encoder.parameters(), lr=0.1, eta=1e-3)
enc_scheduler = lr_scheduler.CosineAnnealingLR(lars, 32, verbose = True)
encoder_params = {
    'model': model_encoder,
    'criterion': contrastive,
    'optimizer': lars,
    'dataloader': enc_loader,
    'dataset_sizes': dataset_sizes,
    'scheduler': enc_scheduler,
    'epochs': 32
}

# %%
enc_history = train_encoder(**encoder_params)
df_enc = pd.DataFrame(enc_history)
df_enc.to_csv("celltype/cell_type_encoder.csv", index = False)

# %%
# all per SupCon paper
dataloaders = {'train': train_loader, 'val': val_loader}
ce = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model_classifier.parameters(), lr = 0.01, weight_decay = 0.005)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.97)

classifier_params = {
    'model': model_classifier,
    'encoder': model_encoder,
    'dataloaders': dataloaders,
    'criterion': ce,
    'optimizer': optimizer,
    'scheduler': scheduler,
    'dataset_sizes': dataset_sizes
}

# %%
history = train_classifier(**classifier_params, epochs = 32)
df_model = pd.DataFrame(history)
df_model.to_csv("cell_type_classifier.csv", index = False)