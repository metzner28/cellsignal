# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from data_loader_normalized import CellSignalDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from helpers_contrastive_v2 import *
import sys
from itertools import chain
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# md_path = "md_test_final.csv"
md_path = '../md_contrastive_train_10views.csv'

test = CellSignalDataset(md_path, transform = None, normalize_file = '../metadata_pixel_stats.csv')
loader = DataLoader(test, batch_size = 40, shuffle = False)

# %%
def get_model_outputs(encoder, classifier, dataloader):

    # preds = []
    embeddings = []
    projections = []
    exp_labels = []
    labels = []

    encoder.eval()
    classifier.eval()

    for input, exp_label, label in tqdm(dataloader):
        
        with torch.set_grad_enabled(False):
            
            get_celltype = lambda ct: 0 if 'HUVEC' in ct else 1 if 'U2OS' in ct else 2 if 'HEPG2' in ct else 3 if 'RPE' in ct else 4
            cell_types = [get_celltype(i) for i in exp_label]
            cell_types = torch.LongTensor(cell_types).to(device)

            # the normalized projection is discarded at test time
            input = input.to(device)
            projection, embedding = encoder(input)
            assert embedding.shape[1] == 512

            # now the embeddings are fed to the classifier
            # output = classifier(embedding, cell_types)
            # softmax = nn.Softmax(dim =  1)
            
            # probs = softmax(output).cpu().detach().numpy()
            embedding = embedding.cpu().detach().numpy()
            projection = projection.cpu().detach().numpy()

            # preds.append(probs)
            projections.append(projection)
            embeddings.append(embedding)
            exp_labels.append(exp_label)
            labels.append(label)

    #preds = np.concatenate(preds, axis = 0)
    embeddings = np.concatenate(embeddings, axis = 0)
    projections = np.concatenate(projections, axis = 0)
    exp_labels = np.array(list(chain(*exp_labels)))
    labels = np.array(list(chain(*labels)))

    return projections, embeddings, exp_labels, labels

# %%
encoder = ContrastiveCellTypeEncoder()
classifier = ContrastiveCellTypeClassifier()

enc = "1204_2v_contrastive/1204_contrastive_encoder_72e.pt"
# cls = "contrastive/contrastive_classifier.pt"

try:
    
    encoder.load_state_dict(
        torch.load(enc)
    )
    encoder = encoder.to(device)

    # classifier.load_state_dict(
    #     torch.load(cls)
    # )
    # classifier = classifier.to(device)

except:
    raise FileNotFoundError("issue with model state dict paths")

# %%
projections, embeddings, exp_labels, labels = get_model_outputs(encoder, classifier, loader)

# %%
model_string = '221206_contrastive_2v_72'

# df_preds = pd.DataFrame(np.column_stack((preds, exp_labels, labels)))
# df_preds.rename(columns = {1139:"experiment", 1140:"label"}, inplace = True)
# df_preds.to_csv(f"contrastive/{model_string}_preds.csv", index = False)
# %%
df_embeddings = pd.DataFrame(np.column_stack((embeddings, exp_labels, labels)))
df_embeddings.rename(columns = {512:"experiment", 513:"label"}, inplace = True)
df_embeddings.to_csv(f"1204_2v_contrastive/{model_string}_embeddings.csv", index = False)

# %%
# df_projection = pd.DataFrame(np.column_stack((projections, exp_labels, labels)))
# df_projection.rename(columns = {128:"experiment", 129:"label"}, inplace = True)
# df_projection.to_csv(f"5vnl_contrastive/{model_string}_projections.csv", index = False)

