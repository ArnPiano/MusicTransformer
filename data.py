import utils
import random
import pickle
import numpy as np
from config import *

#from custom.config import config


class Data:
    def __init__(self, dir_path):
        self.files = list(utils.find_files_by_extensions(dir_path, ['.pickle']))
        self.file_dict = {
            'train': self.files[:int(len(self.files) * 0.8)],
            'eval': self.files[int(len(self.files) * 0.8): int(len(self.files) * 0.9)],
            'test': self.files[int(len(self.files) * 0.9):],
        }
        self._seq_file_name_idx = 0
        self._seq_idx = 0
        pass

    def __repr__(self):
        return '<class Data has "'+str(len(self.files))+'" files>'

    def __len__(self):
        return len(self.files)

    def batch(self, batch_size, length, mode='train'):
        batch_files = random.sample(self.file_dict[mode], k=batch_size)
        batch_data = [ self._get_seq(file, length) for file in batch_files ]
        return np.array(batch_data)  # batch_size, seq_len

    def slide_seq2seq_batch(self, B, L, mode='train'):
        '''
        Returns input sequences until the penultimate value, output sequences without the first value

        :param B: batch size
        :param L: length of the sample/of the model
        :param mode: str, selects the group from the data; it can be 'train', 'eval', or 'test'
        '''

        data = self.batch(B, L+1, mode)
        x = data[:, :-1]
        y = data[:, 1:]

        return x, y


    def _get_seq(self, fname, max_length=None):
        '''
        Returns a section of a sequence from directory 'fname'
        The section is taken at random between the start of the sequence and max_length
        '''
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        if max_length is not None:
            if max_length <= len(data):
                start = random.randrange(0,len(data) - max_length)
                data = data[start:start + max_length]
            else:
                #raise IndexError
                data = np.append(data, eos_token)
                while len(data) < max_length:
                    data = np.append(data, pad_token)
        return data


if __name__ == '__main__':
    from math import sqrt as sqrt
    import pprint
    def count_dict(max_length, data):
        cnt_arr = [0] * max_length
        cnt_dict = {}
        # print(cnt_arr)
        for batch in data:
            for index in batch:
                try:
                    cnt_arr[int(index)] += 1

                except:
                    print(index)
                try:
                    cnt_dict['index-'+str(index)] += 1
                except KeyError:
                    cnt_dict['index-'+str(index)] = 1
        return cnt_arr

    dataset = Data('encoded_in')
    lengths = []
    median = None
    leng = 0
    none = 0
    max_val = -1
    min_val = 300
    for i, fname in enumerate(dataset.files):

        if i == int((len(dataset.files)/2)):
            median = len(dataset._get_seq(fname))
        x = dataset._get_seq(fname)
        if len(x) == 0:
            none +=1
            continue
        else:
            lengths.append(len(x))

        curr_max_val = max(x)
        curr_min_val = min(x)
        max_val = max_val if curr_max_val<max_val else curr_max_val
        min_val = min_val if curr_min_val>min_val else curr_min_val
        t = type(x)

    lengths.sort()
    print(f'there are {len(dataset)} files')
    print(f'of which there are {none} of length 0')
    mean = sum(lengths)/(len(dataset)-none)
    print(f'mean length is {mean}')
    print(f'median is {median}')
    print(f'minimum is {lengths[0]}\nmaximum is {lengths[len(lengths)-1]}')
    diff = lengths
    for d in diff:
        d = (d - mean)**2
    dev = sqrt(sum(diff)/((len(dataset)-none)-1))
    print(f'deviation is {dev}')
    print(f'maximum value in the variables is {max_val}')
    print(f'minimum value in the variables is {min_val}')
    print(f'data type is {t}')
