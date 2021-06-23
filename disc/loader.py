from functools import partial
import numpy as np
import torch
import os
from PIL import Image
from typing import Any, Callable, List, Optional, Union, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

def get_data_loader(root, level, split, batch_size, num_workers, prefetch_factor):
    transforms = Compose([normalize, transform_to_tensor])
    dset = Disc(root, level, split, transform=transforms)

    loader = DataLoader(dset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        prefetch_factor=prefetch_factor,
                        shuffle=True,
                        persistent_workers=True,
                        pin_memory=True)
    return loader

def transform_to_tensor(data_array):
    tensor_array = torch.tensor(data_array)
    # Add channel and batch dim:
    tensor_array = tensor_array.unsqueeze(0)
    return tensor_array

def normalize(data_array):

        assert np.min(data_array) >= 0, "Data type not supported for normalization"
        assert np.max(data_array) <= 255, "Data type not supported for normalization"
        data_array = data_array/255.0
        data_array = data_array*2.0 - 1.0
        return data_array

class Disc(Dataset):
    def __init__(
            self,
            root: str,
            level: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            nts=None,
            balance=False
    ) -> None:
        super(Disc, self).__init__()
        self.root = root 
        self.transform=transform
        self.target_transform=target_transform
        self.split = split

        level_dir = os.path.join(self.root, level)
        self.classes = []
        for c in os.listdir(level_dir):
            try:
                self.classes.append(int(c))
            except:
                pass

        splits = {"train": [0,0.7], 
                  "validation": [0.7,0.8], 
                  "test": [0.8,1], 
                  "all": [0,1]}


        self.samples = []
        for c in self.classes:
            class_dir = os.path.join(level_dir, str(c))
            samples_class = [(os.path.join(class_dir, f), c) for f in os.listdir(class_dir) if f.endswith(".png")] 
            split_0 = int(len(samples_class) * splits[split][0])
            split_1 = int(len(samples_class) * splits[split][1])
            samples_split = samples_class[split_0:split_1]
            self.samples.append(samples_split)

        # Interleave samples
        self.samples = [k for j in zip(*self.samples) for k in j]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample = self.samples[index]
        x = Image.open(sample[0]).convert('L')
        x = np.array(x, dtype=np.float32)
        y = sample[1]
        
        if self.transform is not None:
            x = self.transform(x)

        return x,y 

    def __len__(self) -> int:
        return len(self.samples)
