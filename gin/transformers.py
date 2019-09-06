import torch
from collections import Counter
import numpy as np

class Random(object):
    r"""Adds a random vector to each node feature.

    Args:
        vector_size (int, optional): The size of vector to add. (default: :obj:`10`)
        cat (bool, optional): If set to :obj:`False`, all existing node
            features will be replaced. (default: :obj:`True`)
    """

    def __init__(self, vector_size=100, cat=True):
        self.vector_size = vector_size
        self.cat = cat

    def __call__(self, data):
        x = data.x

        c = torch.randn(size=(data.num_nodes, self.vector_size))

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, c.to(x.dtype).to(x.device)], dim=-1)
        else:
            data.x = c

        return data

    def __repr__(self):
        return '{}(value={})'.format(self.__class__.__name__, self.value)