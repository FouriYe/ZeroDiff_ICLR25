import torch
import torch.utils.data as data

import numpy as np
from PIL import Image

class TestDataset(data.Dataset):

    def __init__(self, img_path, atts, labels, transforms=None, atts_offset=0):
        self.img_path = img_path
        self.labels = torch.tensor(labels).long()
        self.classes = np.unique(labels)
        self.atts = torch.tensor(atts).float()
        self.atts_offset = atts_offset

        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.labels[index]
        att = self.atts[label+self.atts_offset]

        return img, att, label

    def __len__(self):
        return self.labels.size(0)
