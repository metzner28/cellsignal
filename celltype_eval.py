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
data = pd.read_csv("../md_test_final.csv")
# data = data[data["site"] == 1]
# data["path"] = data.apply(lambda df: df["path"].replace("/Volumes/DRIVE/rxrx1/", "/home/ubuntu/cellsignal/"), axis = 1)
# data.to_csv("md_test_final.csv", index = False)

# %%
test = CellSignalDataset("md_test_final.csv")
loader = DataLoader(test, batch_size = 32, shuffle = False)

# %%
def get_model_outputs(model, dataloader):

    preds = []
    embeddings = []
    exp_labels = []
    labels = []

    model.eval()
    
    for input, exp_label, label in tqdm(dataloader):
        
        with torch.set_grad_enabled(False):
            
            input = input.to(device)
            # print(exp_label)
            # exp_label = exp_label.to(device)
            # label = label.to(device)
            
            embeddings, probs = model(input)
            probs = probs.cpu().detach().numpy()
            embeddings = embeddings.cpu().detach().numpy()
        
            preds.append(probs)
            embeddings.append(embeddings)
            exp_labels.append(exp_label)
            labels.append(label)

        break
    
    preds = np.concatenate(preds, axis = 0)
    embeddings = np.concatenate(embeddings, axis = 0)
    exp_labels = np.array(list(chain(*exp_labels)))
    labels = np.array(list(chain(*labels)))

    return preds, exp_labels, labels


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
        output_embedding = nn.ReLU(emb_x)
        fc_final = self.fc(output_embedding)

        softmax = nn.Softmax(dim = 1)
        output_probs = softmax(fc_final)



        return output_embedding, output_probs

try:
    model.load_state_dict(
        torch.load(sys.argv[1])
    )
except:
    raise ValueError("model weights don't exist at specified path")

model = CellTypeModel()
model = model.to(device)

# %%
preds, exp_labels, labels = get_model_outputs(model, loader)

# %%
df_preds = pd.DataFrame(np.column_stack((preds, exp_labels, labels)))
df_preds.rename(columns = {1139:"experiment", 1140:"label"}, inplace = True)

model_string = sys.argv[1].split(".")[-2]
df_preds.to_csv(f"{model_string}_preds.csv", index = False)