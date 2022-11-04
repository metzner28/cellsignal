# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from data_loader_gpu import CellSignalDataset
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
data = pd.read_csv("md_test.csv")
data = data[data["site"] == 1]
data["path"] = data.apply(lambda df: df["path"].replace("/Volumes/DRIVE/rxrx1/", "/home/ubuntu/cellsignal/"), axis = 1)
data.to_csv("md_test.csv", index = False)

# %%
test = CellSignalDataset("md_test.csv")
loader = DataLoader(test, batch_size = 32, shuffle = False)

def get_predictions(model, dataloader):
    
    print('getting predictions')
    preds = []
    exp_labels = []
    true_labels = []

    model.eval()

    for input, exp_label, label in tqdm(dataloader):
        
        with torch.set_grad_enabled(False):
            
            input = input.to(device)
            # exp_label = exp_label.to(device)
            label = label.to(device)
            
            output = model(input)
            print('done')
            _, pred = torch.max(output, 1)
            pred_np = pred.cpu().detach().numpy()
        
            preds.append(pred_np)
            exp_labels.append(exp_label)
            true_labels.append(label)

# %%
model = torchvision.models.resnet18()
trained_kernel = model.conv1.weight
new_conv = nn.Conv2d(6, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
with torch.no_grad():
    new_conv.weight[:] = torch.stack([torch.mean(trained_kernel, 1)] * 6, dim = 1)
model.conv1 = new_conv
model.fc = nn.Linear(model.fc.in_features, 1139)
model = model.to(device)
model.load_state_dict(
    torch.load("resnet18_baseline_fully_trained.pt")
)

# %%
preds, exp_labels = get_predictions(model, loader)
df_predictions = pd.Dataframe({
    'pred_class': preds, 
    'exp_label': exp_labels
})
df_predictions.to_csv("resnet18_baseline_preds.csv", index = False)