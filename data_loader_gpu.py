
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader

# %%
class CellSignalDataset(Dataset):

    def __init__(self, annotations_file, transform = None, target_transform = None):

        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        
        img_location = self.img_labels.iloc[idx, self.img_labels.columns.get_loc("path")]
        img_paths = [img_location + f"{w}.png" for w in range(1,7)]
        imgs = [read_image(img_path) for img_path in img_paths]
        
        
        exp_label = self.img_labels.iloc[idx, self.img_labels.columns.get_loc("experiment")]
        sirna_label = self.img_labels.iloc[idx, self.img_labels.columns.get_loc("sirna_id")]
        
        if self.transform:
            imgs = [self.transform(img) for img in imgs]
        
        if self.target_transform:
            label = self.target_transform(label)
        
        img_final = torch.cat(imgs, dim = 0) / 255
        return img_final, exp_label, sirna_label

def plot_image(image, exp = None, sirna = None, save = False):
        
        if image.shape != (6,512,512):
            raise ValueError("Expects a 6-channel image tensor")
        
        fig, ax = plt.subplots(ncols = 6, figsize = (30,10))
        
        channels = ["Hoechst", "ConA", "Phalloidin", "Syto14", "MitoTracker", "WGA"]
        features = ["nuclei", "cell surface", "actin", "nucleic acids", "mitochondria", "cell membrane"]
        cmaps = ["Blues", "magma", 'Greens', "viridis", "PuRd", "OrRd"]

        for idx, (channel, cmap, feature) in enumerate(zip(channels, cmaps, features)):
            ax[idx].imshow(torch.permute(image[idx:idx+1, :, :], (2,1,0)), cmap = cmap)
            ax[idx].set_title(f"{channel}: {feature}")

        if exp and sirna:
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
            plt.suptitle(f"Experiment: {exp}, siRNA: {sirna}", y = 0.8, size = 'xx-large')
            plt.tight_layout()

            if save:
                plt.savefig(f"{exp}_{sirna}.pdf", format = 'pdf')
            
            plt.show()
        
        else: 
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
            plt.tight_layout()

            if save:
                plt.savefig(f"random_cell.pdf", format = 'pdf')
            
            plt.show()
