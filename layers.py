import utils
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math as m

from config import *

class RelativeGlobalAttention(nn.Module):

    def __init__(self, H, D, L):
        super(RelativeGlobalAttention, self).__init__()
        assert(D % H == 0), "Embedding size D must be divisible by H"
        self.Q_len = None
        self.K_len = None
        self.V_len = None

        self.H = H
        self.D = D
        self.L = L
        self.DH = int(D//H)
        self.E = torch.rand(size=(L, self.DH),requires_grad=False)

        self.Wq = nn.Linear(D,D)
        self.Wk = nn.Linear(D,D)
        self.Wv = nn.Linear(D,D)
        self.fc = nn.Linear(D,D)


    def forward(self, Q, K, V, mask=None, get_weights: bool = False) -> torch.tensor :
        """

        :param Q: tensor of shape (batch_size, L, D)
        :param K: tensor of shape (batch_size, L, D)
        :param V: tensor of shape (batch_size, L, D)
        :param mask: tensor of shape (batch_size, H , L, L)
        :return: tensor of shape (batch_size, length, embedding_dimension)
        """

        Q = self.Wq(Q)
        Q = torch.reshape(Q, (Q.size(0),Q.size(1), self.H, -1 )) # (B, L, H, DH)
        Q = Q.permute(0,2,1,3) # (B, H, L, DH)

        K = self.Wk(K)
        K = torch.reshape(K, (K.size(0), K.size(1), self.H, -1))
        K = K.permute(0, 2, 1, 3)

        V = self.Wv(V)
        V = torch.reshape(V, (V.size(0), V.size(1), self.H, -1))
        V = V.permute(0, 2, 1, 3)

        self.Q_len = Q.size(2)
        self.K_len = K.size(2)
        self.V_len = V.size(2)

        E = self._get_left_embedding(self.Q_len).to(Q.device)
        QE = torch.einsum('bhld,ed->bhle', [Q, E])
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        Kt = K.permute(0,1,3,2) # (B, H, DH, L)
        QK = torch.matmul(Q, Kt)
        logits = QK + Srel
        logits /= m.sqrt(self.DH)
        if mask is not None:
            logits += (mask.to(torch.int64)*-1e9).to(logits.dtype)
        AW = F.softmax(logits, -1) # attention weights
        out = torch.matmul(AW, V) # attention
        out = out.permute(0,2,1,3)
        out = torch.reshape(out, (out.size(0), -1, self.D))
        out = self.fc(out)

        if get_weights:
            return out, AW
        return out

    def _get_left_embedding(self, Q_len) -> torch.tensor:
        """
        Reduces size of E, in case Q and the input have different length
        Shouldn't be needed in this project.

        :param Q_len: int, length of matrix Q
        :return: tensor of shape (L, DH) or (Qlen , DH)
        """

        start = max(0, self.L - Q_len)
        e = self.E[start:, :]
        return e


    def _skewing(self, T: torch.Tensor) -> torch.tensor:
        """
        Skews the tensor to get Srel from QE, as described in the paper Music Transformer.

        :param T:tensor of shape (batch_size, H, DH, DH)
        :return: tensor of shape (batch_size, H, DH, DH)
        """

        padded = F.pad(T, [1, 0]) # pad left column
        reshaped = torch.reshape(padded, shape=[padded.size(0), padded.size(1), padded.size(-1), padded.size(-2)])
        Srel = reshaped[:, :, 1:, :] # slice first row
        if self.K_len > self.Q_len:
            Srel = F.pad(Srel, [0, 0, 0, 0, 0, 0, 0, self.K_len - self.K_len])
        elif self.K_len < self.Q_len:
            Srel = Srel[:, :, :, :self.K_len]

        return Srel

    @staticmethod
    def _qe_masking(qe) -> torch.tensor:
        '''
        Masks tensor qe so that qe[i, j, :, :] is a lower anti-triangular matrix for each i, j.

        :param qe: tensor of shape (B, H, L, L)
        :return: tensor of shape (L, L)

        '''
        mask = utils.sequence_mask(
            torch.arange(qe.size()[-1] - 1, qe.size()[-1] - qe.size()[-2] - 1, -1).to(qe.device),
            qe.size()[-1])
        mask = ~mask.to(mask.device)
        return mask.to(qe.dtype) * qe


class DecoderLayer(nn.Module):
    def __init__(self, D: int, H: int, L: int, rate: float, d: int):
        super(DecoderLayer, self).__init__()
        self.D = D
        self.d = d
        self.rga = RelativeGlobalAttention(H, D, L)

        self.ff1 = nn.Linear(D, self.d)
        self.ff2 = nn.Linear(self.d, D)

        self.norm1 = nn.LayerNorm(self.D, eps=1e-6)
        self.norm2 = nn.LayerNorm(self.D, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x: torch.tensor, mask: Optional[torch.tensor]=None, get_weights: bool = False) -> torch.tensor:
        """

        :param x: tensor of shape (B, L, D)
        :param mask: tensor of shape (B, 1, L, L)
        :return: tensor of shape (B, L, D)
        """
        if get_weights:
            out1, w = self.rga(x, x, x, mask, get_weights)
        else:
            out1 = self.rga(x, x, x, mask, get_weights)
        out1 = self.dropout1(out1)
        out1 = self.norm1(out1+x)

        out2 = F.relu(self.ff1(out1))
        out2 = self.ff2(out2)
        out2 = self.dropout2(out2)
        out2 = self.norm2(out2 + out1)

        if get_weights:
            return out2, w

        return out2


class Decoder(nn.Module):
    def __init__(self, D: int, N: int, L: int, H: int,
                 vocab_size: int = vocab_size, d: Optional[int] = None, rate: float = .1):
        super(Decoder, self).__init__()

        self.D = D
        self.N = N
        self.H = H
        self.L = L
        if d is not None:
            self.d = d
        else:
            self.d = D//2

        self.embedding = nn.Embedding(vocab_size, D)
        self.pos_embedding = utils.DynamicPositionEmbedding(D, L)

        self.EncoderLayers = nn.ModuleList(
            [DecoderLayer (D = self.D, H = self.H, L = self.L, rate = rate, d= self.d)
             for _ in range(self.N)]
        )

        self.dropout = nn.Dropout(rate)

    def forward(self, x: torch.tensor, mask: Optional[torch.tensor] = None, get_weights: bool = False) -> torch.tensor:
        """
        Embeds the Tensor into a D dimensional shape, using both learnable embedding and fixed positional embedding.
        Then it passes the Tensor through the N Decoder Layers.

        :param x: tensor of shape (B, L)
        :param mask: tensor of shape (B, 1, L, L)
        :return: tensor of shape (B, L, D)
        """



        x = self.embedding(x.to(torch.long))
        x *= m.sqrt(self.D)
        x = self.pos_embedding(x)

        if get_weights:
            weights = []
            for i in range(self.N):
                x, w = self.EncoderLayers[i](x, mask, get_weights)
                weights.append(w)
            return x, weights

        for i in range(self.N):
            x = self.EncoderLayers[i](x, mask, get_weights=False)
        return x


if __name__ == '__main__':
    print('Test if each layer is working, and the shape they\'re using')

    x = torch.randint(10, (B, L))
    print('input of the model (tensor from midi):\nshape: ', x.shape, 'type: ', x.dtype, '\n')
    embedding = nn.Embedding(vocab_size, D)
    x = embedding(x)
    print('output of Embedding/input of DynamicPositionEmbedding class:\nshape: ', x.shape, 'type: ', x.dtype, '\n')

    dyn_pos_embedding = utils.DynamicPositionEmbedding(D, L)
    x = dyn_pos_embedding(x)
    print('output of DynamicPositionEmbedding class/input of RelativeGlobalAttention class:\nshape: '
          , x.shape, 'type: ', x.dtype, '\n')

    rga = RelativeGlobalAttention(H, D, L)
    x = rga(x,x,x)
    print('output of  RelativeGlobalAttention class:\nshape:', x.shape, 'type: ', x.dtype, '\n')

    x = torch.rand(B, L, D)
    print('output of DynamicPositionEmbedding class/input of RelativeGlobalAttention class:\nshape: '
          , x.shape, 'type: ', x.dtype, '\n')
    enc = DecoderLayer(D, H, L, rate, d)

    print('input for DecoderLayer class:\nshape: ', x.shape, 'type: ',x.dtype, '\n')
    x = enc(x)
    print('output for DecoderLayer class:\nshape: ', x.shape, 'type: ', x.dtype, '\n')

    nenc = Decoder(D, N, L, H, vocab_size, d, rate)
    x = torch.randint(10, (B, L))
    # x = torch.rand(B, L)
    print('input for Decoder class:\nshape: ', x.shape, 'type: ', x.dtype, '\n', flush=True)
    x = nenc(x)
    print('output for Decoder class:\nshape: ', x.shape, 'type: ', x.dtype, '\n')
