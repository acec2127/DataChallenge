
import os
import random

import numpy as np

import torch

from PIL import Image

from torch.utils.data import Dataset

from utils import FixRandomSeed

import torchvision.transforms as T

#LOAD_TRUNCATED_IMAGES = True

class DatasetLabeled(Dataset):
    def __init__(self, mode, val_ratio, seed, transform) :
        super().__init__()
        self.samples = []
        self.transform = transform

        labelled_idx = np.arange(200)
        idx_split = int((1-val_ratio)*200)
        with FixRandomSeed(seed) :
            np.random.shuffle(labelled_idx)
            if mode == 'train' : 
                labelled_idx = labelled_idx[:idx_split]
            else :
                labelled_idx = labelled_idx[idx_split:]
            for i in labelled_idx :
                self.samples.append((os.path.join('data', f'X_train', f'{i}.png'),
                                    os.path.join('data', f'y_train', f'{i}.png')))
            
    def __getitem__(self, index):
        path_img, path_label = self.samples[index]
        img = Image.open(path_img)
        label = Image.open(path_label)
        seed = random.randint(0, int(2 ** 32))
        return self.transform(img, label, seed)

    def __len__(self):
        return len(self.samples)
    
class DatasetUnlabeled(Dataset):
    def __init__(self, mode, val_ratio, train_val_split_seed, transform) :
        super().__init__()
        self.samples = []
        self.transform = transform
            
        unlabelled_idx = np.arange(200, 1000)
        idx_split = int((1-val_ratio)*800)
        with FixRandomSeed(train_val_split_seed) :
            np.random.shuffle(unlabelled_idx)
            if mode == 'train' : 
                unlabelled_idx = unlabelled_idx[:idx_split]
            else :
                unlabelled_idx = unlabelled_idx[idx_split:]   
        for i in unlabelled_idx:
            self.samples.append(os.path.join('data', f'X_train', f'{i}.png'))

    def __getitem__(self, index):
        transform_seed = int(random.randint(0, int(2 ** 32)))
        path_img = self.samples[index]
        img = Image.open(path_img)
        return *self.transform(img, transform_seed), transform_seed
        
    def __len__(self):
        return len(self.samples)
        
class DatasetTest(Dataset):
    def __init__(self) :
        super().__init__()
        self.samples = []

        for i in range(500):
            self.samples.append(os.path.join('data', f'X_test', f'{i}.png'))
            
    def __getitem__(self, index):
        path_img = self.samples[index]
        img = Image.open(path_img)
        return T.functional.normalize(T.functional.pil_to_tensor(img).to(torch.float), 0, 0.5),\
            path_img

    def __len__(self):
        return len(self.samples)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
