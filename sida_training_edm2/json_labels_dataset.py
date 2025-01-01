# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

"""Distill pretraind diffusion-based generative model using the techniques described in the
paper "Adversarial Score Identity Distillation: Rapidly Surpassing the Teacher in One Step"."""

"""Data loader for dummy images"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
import dnnlib

class ImageDataset(Dataset):
    def __init__(self, path, resolution=64, name='imagenet512', use_labels=True, num_classes=1000, max_size=None,xflip=False):
        super().__init__()
        self.filepath = path
        self.resolution = resolution
        self.name = name
        self.has_labels = use_labels
        self.num_classes = num_classes
        self.max_size = max_size
        self.xflip=xflip

        # Load data
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        self.labels = data['labels']  # List of [filename, label]

        # Limit dataset size if max_size is set
        if self.max_size is not None and self.max_size < len(self.labels):
            self.labels = self.labels[:self.max_size]

        self.labels_array = np.array([label for _, label in self.labels], dtype=np.int64)
        
        print(f"Loaded {len(self.labels)} labels.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Generate a dummy image tensor
        #image = np.random.normal(size=(8, self.resolution, self.resolution)).astype(np.float32)
        image = np.zeros((8, self.resolution, self.resolution), dtype=np.float32)
        return image, self.get_label(idx) 

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = idx #self.labels_array[idx]
        d.xflip = self.xflip
        d.raw_label = self.labels_array[idx].copy()
        return d
    
    def get_label(self, idx):
        # Generate one-hot encoded label dynamically
        label = np.zeros(self.num_classes, dtype=np.float32)
        label[self.labels_array[idx]] = 1.0
        return label.copy()

    