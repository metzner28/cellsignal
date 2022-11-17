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
import sys
from itertools import chain
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# data = pd.read_csv("../md_test_final.csv")
# data = data[data["site"] == 1]
# data["path"] = data.apply(lambda df: df["path"].replace("/Volumes/DRIVE/rxrx1/", "/home/ubuntu/cellsignal/"), axis = 1)
# data.to_csv("md_test_final.csv", index = False)

# %%
test = CellSignalDataset("md_test_final.csv")
loader = DataLoader(test, batch_size = 32, shuffle = False)

# %%
def get_model_outputs(model, dataloader):

    preds = []
    exp_labels = []
    labels = []

    model.eval()
    
    for input, exp_label, label in tqdm(dataloader):
        
        with torch.set_grad_enabled(False):
            
            input = input.to(device)
            # print(exp_label)
            # exp_label = exp_label.to(device)
            # label = label.to(device)
            
            output = model(input)
            softmax = nn.Softmax(dim = 1)
            probs = softmax(output).cpu().detach().numpy()
        
            preds.append(probs)
            exp_labels.append(exp_label)
            labels.append(label)
    
    preds = np.concatenate(preds, axis = 0)
    exp_labels = np.array(list(chain(*exp_labels)))
    labels = np.array(list(chain(*labels)))

    return preds, exp_labels, labels


# %%
model = torchvision.models.resnet18()
trained_kernel = model.conv1.weight
new_conv = nn.Conv2d(6, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
with torch.no_grad():
    new_conv.weight[:] = torch.stack([torch.mean(trained_kernel, 1)] * 6, dim = 1)
model.conv1 = new_conv
model.fc = nn.Linear(model.fc.in_features, 1139)
# model.load_state_dict(
#          torch.load("../results/resnet18_baseline_mdfinal.pt")
# )

try:
    model.load_state_dict(
        torch.load(sys.argv[1])
    )
except:
    raise ValueError("model weights don't exist at specified path")

model = model.to(device)

# %%
preds, exp_labels, labels = get_model_outputs(model, loader)

# %%
df_preds = pd.DataFrame(np.column_stack((preds, exp_labels, labels)))
df_preds.rename(columns = {1139:"experiment", 1140:"label"}, inplace = True)

model_string = sys.argv[1].split("/")[-1].split(".")[0]
df_preds.to_csv(f"{model_string}_preds.csv", index = False)