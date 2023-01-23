import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from utils.utils import *
import pandas as pd


class ExperimentDataset(Dataset):
    def __init__(self, path, data, targets):
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        self.data, self.targets = data, targets
        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y, idx


def get_unsw_nb15(is_binary=True):
    file_dir = "./data/unsw-nb15/raw/data_set"
    assert os.path.isdir(file_dir), "There is no datasets"
    all_data = _read_all_csv(file_dir)
    if is_binary:
        del all_data[47]
    else:
        del all_data[48]
        events = all_data[47].value_counts()
        value_map: dict = dict((v, i + 1) for i, v in enumerate(events.index))
        all_data[47].replace(value_map, inplace=True)
        all_data[47] = all_data[47].fillna(0)
        all_data[47] = all_data[47].astype(int)

    all_data[1].replace({'0x000b': 11, '0x000c': 12, '-': 0}, inplace=True)
    all_data[3].replace({'0xc0a8': 49320, '-': 0, '0xcc09': 52233, '0x20205321': 538989345}, inplace=True)
    all_data[39].replace({' ': 0}, inplace=True)
    all_data[1] = all_data[1].astype(int)
    all_data[3] = all_data[3].astype(int)
    all_data[39] = all_data[39].astype(int)

    tran_protocols = all_data[4].value_counts()
    value_map = dict((v, i) for i, v in enumerate(tran_protocols.index))
    all_data[4].replace(value_map, inplace=True)
    all_data[4] = all_data[4].astype(int)

    dep_protocols = all_data[5].value_counts()
    value_map = dict((v, i) for i, v in enumerate(dep_protocols.index))
    all_data[5].replace(value_map, inplace=True)
    all_data[5] = all_data[5].astype(int)

    service = all_data[13].value_counts()
    value_map = dict((v, i) for i, v in enumerate(service.index))
    all_data[13].replace(value_map, inplace=True)
    all_data[13] = all_data[13].astype(int)

    all_data[2].replace(_ip_addresses_convert_nums(all_data[2].unique()), inplace=True)
    # all_data[2] = (all_data[2] - 0) / 4294967295
    # all_data[2] = all_data[2].astype(float)
    all_data[2] = all_data[2].astype(int)

    all_data[0].replace(_ip_addresses_convert_nums(all_data[0].unique()), inplace=True)
    # all_data[0] = (all_data[0] - 0) / 4294967295
    # all_data[0] = all_data[0].astype(float)
    all_data[0] = all_data[0].astype(int)

    all_data[37] = all_data[37].fillna(37)
    all_data[38] = all_data[38].fillna(5)
    all_data[37] = all_data[37].astype(int)
    all_data[38] = all_data[38].astype(int)

    label = all_data.iloc[:, -1]
    text = all_data.iloc[:, :-1]
    # text_norm = (text - text.min()) / (text.max() - text.min())
    # text_norm[0] = all_data[0]
    # text_norm[2] = all_data[2]
    return torch.tensor(np.array(text), dtype=torch.float32), torch.tensor(np.array(label))


def get_n_baiot(is_binary=True):
    file_dir = "./data/n-baiot/raw/data_set"
    assert os.path.isdir(file_dir), "There is no datasets"
    all_data = _read_all_csv(file_dir)
    del all_data[0]
    label = all_data.iloc[:, -1]
    text = all_data.iloc[:, :-1]
    # text_norm = (text - text.min()) / (text.max() - text.min())
    return torch.tensor(np.array(text), dtype=torch.float32), torch.tensor(np.array(label))


def _ip_addresses_convert_nums(ips):
    result = dict()
    for pnt in range(len(ips)):
        x = ips[pnt]
        z = 0
        parts = x.split('.')
        z = (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
        result[x] = z
    return result


def _read_all_csv(file_dir):
    all_csv_list = os.listdir(file_dir)
    all_data = None
    for csv_file in all_csv_list:
        data = pd.read_csv(os.path.join(file_dir, csv_file), header=None)
        if csv_file == all_csv_list[0]:
            all_data = data
        else:
            all_data = pd.concat([all_data, data], ignore_index=True)
    return all_data