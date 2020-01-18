import math

import torch


def split_dataset(dataset, test_size):
    train_size = math.floor(dataset.__len__() * test_size)
    test_size = dataset.__len__() - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])