# utils/collate.py

import torch
from typing import List, Tuple, Any

def collate_latent_batch(batch):
    mixes, _, lat_paths, starts = zip(*batch)
    return torch.stack(mixes, 0), lat_paths, torch.tensor(starts)

'''
def collate_latent_batch(
    batch: List[Tuple[torch.Tensor, Any, Tuple[str, str]]]
):
    """
    Stacks mixture tensors, leaves lat_paths as a list-of-tuples.
    `srcs` is ignored (can be None).

    Returns
    -------
    mix_stack : torch.Tensor  [B, 1, T]
    lat_paths : List[Tuple[str,str]]  length B
    """
    mixes, _, lat_paths = zip(*batch)   # unpack
    mix_stack = torch.stack(mixes, 0)   # default_collate on tensors
    return mix_stack, lat_paths
'''
