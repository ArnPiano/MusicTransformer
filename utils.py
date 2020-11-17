import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math as m
import torchvision
from config import *


def find_files_by_extensions(root, exts=[]):
    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False
    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)

def remove_out_of_range_from_seq(seq: np.array, vocab_size: int = pad_token):

    removed = seq[ (seq<0) | (seq > vocab_size) ]
    seq = seq[ (0 <= seq) &  (seq < vocab_size)]
    print(f'removed {len(removed)} elements:\n', removed, flush=True)
    return seq

def sequence_mask(length, max_length=None):
    """Tensorflow sequence_mask"""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def get_masked_with_pad_tensor(L, trg, pad_token=388):
    """
    Masks the upper right triangular part of the last two dimensions of a tensor (upper triangular matrix).
    It masks also every padding token's occurrence.

    :param L: int, size of target tensor
    :param trg: tensor of shape (B, L)
    :param pad_token: int, pad token
    :return: look_ahead_mask: tensor of shape (B, 1, L, L)
    """
    trg = trg[:, None, None, :]
    trg_pad_tensor = torch.ones_like(trg).to(trg.device.type) * pad_token
    dec_trg_mask = trg == trg_pad_tensor
    # boolean reversing i.e) True * -1 + 1 = False
    seq_mask = ~sequence_mask(torch.arange(1, L+1).to(trg.device), L)
    # look_ahead_mask = torch.max(dec_trg_mask, seq_mask)
    look_ahead_mask = dec_trg_mask | seq_mask
    return look_ahead_mask


class DynamicPositionEmbedding(torch.nn.Module):

    def __init__(self, embedding_dim, max_seq=2048):
        super().__init__()
        embed_sinusoid_list = np.array([[
            [
                m.sin(
                    pos * m.exp(-m.log(10000) * i/embedding_dim) *
                    m.exp(m.log(10000)/embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]])
        self.positional_embedding = embed_sinusoid_list

    def forward(self, x) -> torch.tensor:
        '''
        Adds to x a positional embedding, as in 'Self-Attention with Relative Position Representations'.

        :param x: tensor of shape [batch_size, L, D]
        :return: tensor of shape [batch_size, L, D]
        '''
        x = x + torch.from_numpy(self.positional_embedding[:, :x.size(1), :]).to(x.device, dtype=x.dtype)
        return x


class NoamOptimizer:
    '''
    Optim wrapper implementing 'rate' and 'step'
    This scheduler is mentioned in 'Attention is all you need'
    '''
    def __init__(self, D, factor=1, warmup_steps=4000, optimizer=None):
        super(NoamOptimizer, self).__init__()

        self.D = D
        self.factor = factor
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer

        self._step = 0
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** -1.5)

        formula = self.D ** (-0.5) * min(arg1, arg2)

        return self.factor * formula

    def zero_grad(self):
        self.optimizer.zero_grad()


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
    x: a Tensor with shape [..., m]
    n: an integer.
    Returns:
    a Tensor with shape [..., n, m/n]
    """
    x_shape = x.size()
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return torch.reshape(x, x_shape[:-1] + (n, m // n))


def attention_image_summary(name, attn, step=0, writer=None):

    """
    Compute color image summary.
    Args:
    attn: a Tensor with shape [batch, num_heads, query_length, memory_length]
    image_shapes: optional tuple of integer scalars.
      If the query positions and memory positions represent the
      pixels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, memory_rows, memory_cols).
      If the query positions and memory positions represent the
      pixels x channels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, query_channels,
         memory_rows, memory_cols, memory_channels).
    """

    num_heads = attn.size(1)
    # [batch, query_length, memory_length, num_heads]
    image = attn.permute(0, 2, 3, 1)
    image = torch.pow(image, 0.2)  # for high-dynamic-range
    # Each head will correspond to one of RGB.
    # pad the heads to be a multiple of 3
    image = F.pad(image, [0,  -num_heads % 3, 0, 0, 0, 0, 0, 0,])
    image = split_last_dimension(image, 3)
    image = image.max(dim=4).values
    grid_image = torchvision.utils.make_grid(image.permute(0, 3, 1, 2))
    writer.add_image(name, grid_image, global_step=step, dataformats='CHW')
