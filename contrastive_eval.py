# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from data_loader_gpu import CellSignalDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from helpers_contrastive import *
import sys
from itertools import chain
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
md_path = "md_test_final.csv"

test = CellSignalDataset(md_path, transform = None)
loader = DataLoader(test, batch_size = 32, shuffle = False)

# %%
def get_model_outputs(encoder, classifier, dataloader):

    preds = []
    embeddings = []
    exp_labels = []
    labels = []

    encoder.eval()
    classifier.eval()

    for input, exp_label, label in tqdm(dataloader):

        with torch.set_grad_enabled(False):
            
            get_celltype = lambda ct: 0 if 'HUVEC' in ct else 1 if 'U2OS' in ct else 2 if 'HEPG2' in ct else 3 if 'RPE' in ct else 4
            cell_types = [get_celltype(i) for i in exp_labels]
            cell_types = torch.LongTensor(cell_types).to(device)

            # the normalized projection is discarded at test time
            input = input.to(device)
            _, embedding = encoder(input, cell_types)
            assert embedding.shape[1] == 512

            # now the embeddings are fed to the classifier
            output = classifier(embedding)
            softmax = nn.Softmax(dim =  1)
            
            probs = softmax(output).cpu().detach().numpy()
            embedding = embedding.cpu().detach().numpy()

            preds.append(probs)
            embeddings.append(embeddings)
            exp_labels.append(exp_label)
            labels.append(label)

    preds = np.concatenate(preds, axis = 0)
    embeddings = np.concatenate(embeddings, axis = 0)
    exp_labels = np.array(list(chain(*exp_labels)))
    labels = np.array(list(chain(*labels)))

    return preds, embeddings, exp_labels, labels

# %%
encoder = ContrastiveCellTypeEncoder()
classifier = ContrastiveCellTypeClassifier()

try:
    
    encoder.load_state_dict(
        torch.load(sys.argv[1])
    )
    encoder = encoder.to(device)

    classifier.load_state_dict(
        torch.load(sys.argv[2])
    )
    classifier = classifier.to(device)

except:
    raise FileNotFoundError("issue with model state dict paths")

# %%
preds, embeddings, exp_labels, labels = get_model_outputs(encoder, classifier, loader)

# %%
model_string = 'contrastive'

df_preds = pd.DataFrame(np.column_stack((preds, exp_labels, labels)))
df_preds.rename(columns = {1139:"experiment", 1140:"label"}, inplace = True)
df_preds.to_csv(f"{model_string}_preds.csv", index = False)

df_embeddings = pd.DataFrame(np.column_stack((embeddings, exp_labels, labels)))
df_embeddings.rename(columns = {512:"experiment", 513:"label"}, inplace = True)
df_preds.to_csv(f"{model_string}_embeddings.csv", index = False)





            


