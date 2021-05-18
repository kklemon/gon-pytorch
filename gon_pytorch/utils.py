import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset
from gon_pytorch import modules


def get_block_factory(activation='siren', bias=True):
    if activation == 'siren':
        return modules.SirenBlockFactory(nn.Linear, bias=bias)
    if activation == 'relu':
        return modules.LinearBlockFactory(nn.Linear, activation_cls=nn.ReLU, bias=bias)
    if activation == 'leaky_relu':
        return modules.LinearBlockFactory(nn.Linear, activation_cls=lambda: nn.LeakyReLU(0.2), bias=bias)
    if activation == 'swish':
        return modules.LinearBlockFactory(nn.Linear, activation_cls=modules.Swish, bias=bias)
    raise ValueError(f'Unknown activation {activation}')


def get_xy_grid(width, height):
    x_coords = np.linspace(-1, 1, width, endpoint=False)
    y_coords = np.linspace(-1, 1, height, endpoint=False)

    xy_grid = torch.tensor(
        np.stack(np.meshgrid(x_coords, y_coords), -1)
    ).unsqueeze(0).float().contiguous()

    return xy_grid


class NoLabelWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx][0]

    def __len__(self):
        return len(self.dataset)
