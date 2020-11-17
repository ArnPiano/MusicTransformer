from config import *
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss, _Loss

class CategoricalAccuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Y: torch.Tensor, O: torch.Tensor):

        Y = Y.softmax(-1).argmax(-1)
        acc = Y.long() == O.long()
        acc.to(torch.float)

        return acc.sum().true_divide( acc.numel())


class TransformerLoss(CrossEntropyLoss):

    __constants__ = ['vocab_size', 'ignore_index', 'reduction']

    def __init__(self,weight: Optional[torch.tensor] =None, vocab_size: int =vocab_size , size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super().__init__(weight=weight, size_average=size_average,ignore_index=ignore_index,reduce=reduce, reduction=reduction)
        self.vocab_size = vocab_size

    def forward(self, input: torch.tensor, target: torch.tensor) -> torch.tensor:

        input = input.view(-1, self.vocab_size)
        target = target.view(-1).to(torch.long)

        return super().forward(input, target)


class SmoothCrossEntropyLoss(_Loss):
    """
    https://arxiv.org/abs/1512.00567
    """
    __constants__ = ['label_smoothing', 'vocab_size', 'ignore_index', 'reduction']

    def __init__(self, label_smoothing=label_smoothing, vocab_size=vocab_size, ignore_index=-100, reduction='mean', is_logits=True):
        assert 0.0 <= label_smoothing <= 1.0
        super().__init__(reduction=reduction)

        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.input_is_logits = is_logits

    def forward(self, input: torch.tensor, target: torch.tensor):
        """

        :param input: tensor of shape (B, L, vocab_size)
        :param target: tensor of shape (B, vocab_size)
        :return:
        """
        mask = (target == self.ignore_index).unsqueeze(-1)
        q = F.one_hot(target.long(), self.vocab_size).type(torch.float32)
        u = 1.0 / self.vocab_size
        q_prime = (1.0 - self.label_smoothing) * q + self.label_smoothing * u
        q_prime = q_prime.masked_fill(mask, 0)

        # print(q_prime.dtype, 'qprime dtype')
        ce = self.cross_entropy_with_logits(q_prime, input).to(torch.float32)
        if self.reduction == 'mean':
            lengths = torch.sum(target != self.ignore_index)
            return ce.sum() / lengths
        elif self.reduction == 'sum':
            return ce.sum()
        else:
            raise NotImplementedError

    def cross_entropy_with_logits(self, p, q):

        return -torch.sum(p * (q - q.logsumexp(dim=-1, keepdim=True)), dim=-1).to(torch.float32)


if __name__ == '__main__':
    from data import Data
    from model import MusicTransformer

    accuracy = CategoricalAccuracy()
    SCEloss = SmoothCrossEntropyLoss(label_smoothing=label_smoothing)

    dataset = Data(data_dir)
    x, y = dataset.slide_seq2seq_batch(B, L)
    x = torch.from_numpy(x).contiguous().to(device, non_blocking=True, dtype=torch.int)
    y = torch.from_numpy(y).contiguous().to(device, non_blocking=True, dtype=torch.int)

    model = MusicTransformer(D, L, N, H)
    model.train()
    x = model(x)

    print(x.shape, x.dtype)

    loss = SCEloss(x.to(y.device), y)
    print(loss.shape, loss.dtype, loss.item())
    print(loss.item())


    acc = accuracy(x.to(y.device), y)
    print(acc.shape, acc.dtype, acc.item())

    criterion = TransformerLoss()

    print(x.shape, 'x shape', y.shape, 'y shape')
    loss = criterion(x, y)
    print(loss.shape, loss.dtype, loss.item())
