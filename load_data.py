import torch
from torch.utils import data
import scipy.io as sio
import scipy.signal as sig
import numpy as np
import random


def load_data_isrucs3(idx_list, channel, batch_size, window_size,
                      oversampling=False, shuffle=False, filepath='./dataset', loader=True):
    data_sequence, label_sequence = None, None
    for idx in idx_list:
        assert 1 <= idx <= 10
        filepath_data = filepath + '/data/subject' + str(idx) + '.mat'
        filepath_label = filepath + '/label/' + str(idx) + '-Label.mat'
        data_mat = sio.loadmat(filepath_data)
        label_mat = sio.loadmat(filepath_label)
        _, _, Zxx = sig.stft(data_mat[channel], 200, 'hann', 256)
        data_subject = torch.tensor(np.abs(Zxx), dtype=torch.float32)
        label_subject = torch.tensor(label_mat['label'], dtype=torch.int64).view(-1)
        segments = data_subject.shape[0] // window_size
        for seg in range(segments):
            data_seq = data_subject[seg * window_size: (seg + 1) * window_size].clone()
            label_seq = label_subject[seg * window_size: (seg + 1) * window_size].clone()
            if (data_sequence is None) or (label_sequence is None):
                data_sequence = torch.unsqueeze(data_seq, dim=0)
                label_sequence = torch.unsqueeze(label_seq, dim=0)
            else:
                data_sequence = torch.cat((data_sequence, torch.unsqueeze(data_seq, dim=0)), dim=0)
                label_sequence = torch.cat((label_sequence, torch.unsqueeze(label_seq, dim=0)), dim=0)
            if oversampling:
                bias = window_size // 2
                if (seg + 1) * window_size + bias <= data_subject.shape[0]:
                    data_seq = data_subject[seg * window_size + bias: (seg + 1) * window_size + bias].clone()
                    label_seq = label_subject[seg * window_size + bias: (seg + 1) * window_size + bias].clone()
                    data_sequence = torch.cat((data_sequence, torch.unsqueeze(data_seq, dim=0)), dim=0)
                    label_sequence = torch.cat((label_sequence, torch.unsqueeze(label_seq, dim=0)), dim=0)
    if loader:
        dataset = data.TensorDataset(data_sequence, label_sequence)
        return data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=4)
    else:
        return data_sequence, label_sequence


default_tasks_list = ['F3_A2', 'F4_A1', 'O1_A2', 'O2_A1', 'LOC_A2', 'ROC_A1']


class Continuum:
    def __init__(self, tasks_list=None):
        self.train_list = [i for i in range(1, 11)]
        random.shuffle(self.train_list)
        self.valid_idx, self.test_idx = self.train_list[0], self.train_list[1]
        self.train_list.remove(self.valid_idx)
        self.train_list.remove(self.test_idx)
        '''
        print(f'valid idx: {self.valid_idx}, test idx: {self.test_idx}, train list: {self.train_list}')
        '''
        if tasks_list is None:
            self.tasks_list = default_tasks_list
        else:
            self.tasks_list = tasks_list
        self.tasks_num = len(self.tasks_list)

    def get_train(self):
        return self.train_list

    def get_valid(self):
        return [self.valid_idx]

    def get_test(self):
        return [self.test_idx]

    def __getitem__(self, item):
        assert 0 <= item < self.tasks_num
        return self.tasks_list[item]


if __name__ == '__main__':
    data_mat = sio.loadmat('./dataset/data/subject1.mat')
    print(data_mat)
