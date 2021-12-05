import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import glob

seed_job = 2021

from pathlib import Path
from PIL import Image
import logging
import matplotlib.pyplot as plt

class PanDataset(Dataset):
    def __init__(self, 
                 img_dir: str,
                 size_img: int = 224,
                 extension: str = 'png',
                 transform=None,
                 augmentation=None):
        
        
        assert 32 <= size_img <= 2048, 'Size selection is not appropriate, pick between 32 and 2048'
        self.size_img = size_img
        
        lst_tiles = [f for f in glob.glob(img_dir + "/*." + extension)]
        lst_class = [f.split('__')[-1].split('.')[0] for f in lst_tiles]
        lst_class = [''.join(filter(lambda x: not x.isdigit(), c)) for c in lst_class]
        cleaned = [c != "noise" for c in lst_class]
        lst_class = [lst_class[i] for i, flag in enumerate(cleaned) if flag]
        
        self.img_dir = [lst_tiles[i] for i, flag in enumerate(cleaned) if flag]
        self.ids = lst_class
        
        if not self.ids:
            raise RuntimeError(f'No Input Files in {img_dir}!')
        logging.info(f'Creating Dataset: {len(self.ids)} examples...')
    
    def __len__(self):
        return len(self.ids)
    
    @classmethod
    def preprocess(cls, img, size_img, mask):
        w, h = img.size
        w_sized, h_sized = size_img, size_img
        assert w_sized > 32 and h_sized > 32, 'Requesed Image Size is too small!'
        img = img.resize((w_sized, h_sized))
        img_ndarray = np.asarray(img)
        
        if img_ndarray.ndim == 2 and not mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
            img_ndarray = img_ndarray / 255
        elif not mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))
            img_ndarray = img_ndarray / 255
        
        return img_ndarray
    
    @classmethod
    def loader(cls, filename: str):
        return Image.open(filename)
    
    def __getitem__(self, idx: int):
        name_cls = self.ids[idx]
        fname_img = self.img_dir[idx]

        img = self.loader(fname_img)
        ann = {"Stroma":0, "normal":1, "pannet":2}[name_cls]
        
        img = self.preprocess(img, self.size_img, mask=False)
        
        return {'image': torch.as_tensor(img.copy()).float().contiguous(),
               'ann': ann, 'name_img': fname_img}
        
class WsiDataset(Dataset):
    def __init__(self, 
                 img_dir: str,
                 size_img: int = 224,
                 extension: str = 'png',
                 transform=None,
                 augmentation=None):
        
        
        assert 32 <= size_img <= 2048, 'Size selection is not appropriate, pick between 32 and 2048'
        self.size_img = size_img
        
        lst_tiles = [f for f in glob.glob(img_dir + "/*." + extension)]
        
        self.img_dir = lst_tiles
        
        if not self.img_dir:
            raise RuntimeError(f'No Input Files in {img_dir}!')
        logging.info(f'Creating Dataset: {len(self.img_dir)} examples...')
    
    def __len__(self):
        return len(self.img_dir)
    
    @classmethod
    def preprocess(cls, img, size_img, mask):
        w, h = img.size
        w_sized, h_sized = size_img, size_img
        assert w_sized > 32 and h_sized > 32, 'Requesed Image Size is too small!'
        img = img.resize((w_sized, h_sized))
        img_ndarray = np.asarray(img)
        
        if img_ndarray.ndim == 2 and not mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
            img_ndarray = img_ndarray / 255
        elif not mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))
            img_ndarray = img_ndarray[:3,:,:]
            img_ndarray = img_ndarray / 255
        
        return img_ndarray
    
    @classmethod
    def loader(cls, filename: str):
        return Image.open(filename)
    
    def __getitem__(self, idx: int):
        fname_img = self.img_dir[idx]

        img = self.loader(fname_img)
        
        img = self.preprocess(img, self.size_img, mask=False)
        
        return {'image': torch.as_tensor(img.copy()).float().contiguous(), 'name_img': fname_img}