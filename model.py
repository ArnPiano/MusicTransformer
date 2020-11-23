import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter
from data import Data
from midi_processor.processor import decode_midi
import utils
from typing import Optional
import layers
from config import *

from tqdm import tqdm

class MusicTransformer(nn.Module):

    def __init__(self, D: int, L: int, N: int, H: int, d: Optional[int] = None,
                 writer=None, dist =True, rate: float = .1, vocab_size: int =vocab_size):
        """

        :param D: Model dimension/Embedding dimension
        :param L: Fixed length of the input
        :param N: Number of stacked Decoder layers
        :param H: Number of Heads in the attention
        :param d: Dimension of the linear layer at the end of each Decoder
        :param writer: SummaryWriter
        :param dist: bool, for the generation process; if True, uses generated distribution, if False uses argmax of distribution
        :param rate: Dropout rate
        :param vocab_size: size of the vocabulary
        """

        super().__init__()
        self.infer = False

        self.D = D
        self.L = L
        self.N = N
        self.H = H
        if d is not None:
            self.d = d
        else:
            self.d = D//2

        self.vocab_size = vocab_size

        self.writer = writer
        self.dist = dist
        self.Decoder = layers.Decoder(self.D, self.N, self.L, self.H, d = self.d, vocab_size=self.vocab_size, rate=rate)
        self.fc = nn.Linear(self.D, self.vocab_size)

    def forward(self, x: torch.tensor, length: Optional[int] = None) -> torch.tensor:

        """
        If in training mode it generates a tensor of predictions
        In infer mode it generates a tensor of shape (1, n), to be encoded into midi.

        :param x: tensor of shape (B, L)
        :param length: int, length of tokens added to generated incipit
        :return: tensor of shape (B, L, vocab_size) or shape (1, n+length), n a positive integer.
        """

        #  It is training or it is evaluating
        if self.training or not self.infer:
            mask = utils.get_masked_with_pad_tensor(self.L, x)
            #  It is training
            if self.training:
                x = self.Decoder(x, mask)
                x = self.fc(x)
                return x.contiguous()

            #  It is evaluating
            else:
                x, weights = self.Decoder(x, mask, get_weights=True)
                x = self.fc(x)
                return x.contiguous(), [weight.contiguous() for weight in weights]

        #  It is generating
        else:
            return self.generate(x, length)

    def generate(self, x: torch.tensor , length: int, writer: SummaryWriter = None) -> torch.tensor:

        """
        Generates 'length' tokens and concatenates them to a copy of the input tensor 'x'.

        :param x: tensor of shape (1, n), n a positive integer
        :param length: int, length of output to be generated
        :param writer: SummaryWriter
        :return: tensor of shape (composition_length)
        """

        decoded = x
        result = x
        idx = 0
        for _ in tqdm(range(length)):
            if decoded.size(1) >= 2:
                decoded = decoded[:, 1:]
            res = self.Decoder(decoded, None, False)
            res = self.fc(res)
            res = res.softmax(-1)
            if writer:
                writer.add_image('logits', res, global_step = idx)
            if dist:
                pdf = dist.OneHotCategorical(probs=res[:,-1])
                res = pdf.sample().argmax(-1).unsqueeze(-1)
                decoded = torch.cat((decoded, res), dim=-1)
                result = torch.cat((result, res), dim=-1)
            idx += 1

        # get rid of additional dimension
        result = result[0]
        result = result.cpu()
        
        return result

    def infer_mode(self, on: bool = True) -> None:

        """
        Sets the MusicTransformer instance to inference mode by default.
        If a False boolean is passed it sets the instance to training mode.

        :param on: bool
        :return: None
        """

        self.infer = on
        if on:
            self.eval()
        else:
            self.train()


def to_midi(model: MusicTransformer, x: torch.tensor, directory: str, filename: str, length: int) -> bool:

    """
    Sets the model to inference mode and tries to generate a tensor of length (n+ length), n a positive integer.
    Generates the file 'filename'.mid and saves it into 'directory'.

    :param model: MusicTransformer
    :param x: tensor of shape (1, L)
    :param directory: where to save the midi file
    :param filename: midi filename
    :param length: length of the composition
    :return: bool indicating success or failure of the file generation
    """

    model.infer_mode()
    with torch.no_grad():
        z = model(x, length)
        z = z.numpy()
        z = utils.remove_out_of_range_from_seq(z)

        try:
            decode_midi(z, file_path=(directory + f'/{filename}.mid'))
        except ValueError:
            return False

    return True


def test_composition(dataset: Data, model: MusicTransformer, length: int,
                     data_idx: int = 400, data_len: int = 64,
                     n_attempts: int = number_of_trials_before_giving_up, directory: str = midi_out_dir,
                     filename: str = 'test') -> None:
    """
    REMEMBER TO USE .infer_mode(False) after testing.
    Use to test composition

    :param dataset: Dataset from which to extract an example
    :param model: Transformer model, or any other seq2seq model producing a tensor of size (1, D, vocab_size)
    :param length: length of the composition
    :param data_idx: Which data to access
    :param data_len: Length of the incipit
    :param n_attempts: Number of attempts before failing
    :param directory: Directory where midi file is saved
    :param filename: Name of the midi file
    :return: None
    """

    if length is not None:
        incipit = dataset._get_seq(dataset.files[data_idx])[:data_len]
    else:
        incipit = torch.tensor([[12*4+1]])

    incipit = torch.tensor(incipit).contiguous().to(device, non_blocking=True, dtype=torch.int)
    incipit = incipit.unsqueeze(0)

    print('\nGenerating music...\n', flush=True)

    for i in range(n_attempts):
        print(f'attempt n.{i+1}', flush=True)
        done = to_midi(model, incipit, directory, filename, length)
        if done:
            print(f'\ncomposition successful at attempt n.{i+1}\n\n')
            break
        else:
            print('\nComposition unsuccessful: data not in range...\n')


if __name__ == '__main__':
    from data import Data
    print('Testing the model\'s capabilities of producing the desired input and generating a midi file' )

    model = MusicTransformer(D, L, N, H)

    x = torch.rand(B, L)
    print('input of the model (tensor from midi):\nshape: ', x.shape, 'type: ', x.dtype, '\n', flush=True)
    y = model(x)
    print('output of the model:\nshape: ', y.shape, 'type: ', y.dtype, '\n', flush=True)

    model.infer_mode()
    incipit = torch.tensor([[12*4+1]])
    z = model(incipit, composition_length)

    print('output generated from the model :\nshape: ', z.shape, 'type: ', z.dtype, '\n', flush=True)

    print('testing data composition', flush=True)

    dataset = Data(pickle_dir)

    test_composition(dataset, model, composition_length,
                     data_idx=400, data_len=50,
                     n_attempts=number_of_trials_before_giving_up, directory=midi_out_dir,
                     filename='test')
