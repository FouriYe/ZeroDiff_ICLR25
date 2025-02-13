from __future__ import division

import torch
from torchvision import transforms
import numpy as np
import numpy as np
from scipy import linalg
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array

def data_transform_fun(name, size=224):
    name = name.strip().split('+')
    name = [n.strip() for n in name]
    transform = []

    if 'resize_random_crop' in name:
        transform.extend([
            transforms.Resize(int(size * 8. / 7.)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(0.5)
        ])
    elif 'resize_center_crop' in name:
        transform.extend(
            transforms.Resize(size),
            transforms.CenterCrop(size),
        )
    elif 'resize_only' in name:
        transform.extend([
            transforms.Resize((size, size)),
        ])
    elif 'resize' in name:
        transform.extend([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(0.5)
        ])
    else:
        transform.extend([
            transforms.Resize(size),
            transforms.CenterCrop(size)
        ])

    if 'colorjitter' in name:
        transform.extend(
            transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.2)
        )

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transform.extend([transforms.ToTensor(), normalize])
    transform = transforms.Compose(transform)
    return transform

def patchCrop(img, size, lam):
    n,c,w,h = img.shape

    cut_rate = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rate)
    cut_h = np.int(h * cut_rate)

    # uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    patch = img[:,:,bbx1:bbx2,bby1:bby2]
    torch_resize = transforms.Resize(int(size * 8. / 7.))
    patch = torch_resize(patch)
    return patch
