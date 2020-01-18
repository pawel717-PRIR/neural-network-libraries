import logging
from abc import ABC, abstractmethod
from timeit import default_timer as timer
import torch
from sklearn.model_selection import KFold
from torch import nn as nn
import torch.nn.functional as F

from prir.pytorch.data.MaskedSampler import MaskedSampler
from prir.pytorch.data.dataset import split_dataset


class NetworkModel(ABC):
    def __init__(self,
                 device='cuda:0',
                 train_to_test_split_ratio=0.3,
                 batch_size=128,
                 n_epochs=12):

        self.train_to_test_split_ratio = train_to_test_split_ratio
        self.batch_size_train = batch_size
        self.n_epochs = n_epochs
        self.__init_class_fields__()
        self.batch_size_test = batch_size
        self.log_interval = batch_size
        self.device = device
        self.__init_network__()

    @abstractmethod
    def __init_class_fields__(self):
        self.num_classes = None
        self.learning_rate = None
        self.dataset = None

    @abstractmethod
    def __init_network__(self):
        self.network = None
        self.optimizer = None

    def reset(self):
        self.__init_network__()
        return self

    def set_device(self, device):
        self.device = device
        self.network.to(self.device)

    def run_train(self):
        train_set, test_set = split_dataset(self.dataset, test_size=self.train_to_test_split_ratio)
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size_test,
                                                   shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=self.batch_size_test,
                                                  shuffle=False)
        logging.info('''
        ==============================================================
        {} device: {} TRAIN START
        --------------------------------------------------------------
        '''.format(self.__class__.__name__, self.device))
        start = timer()
        self.train_loop(train_loader=train_loader, test_loader=test_loader)
        end = timer()
        print('''
        --------------------------------------------------------------
        {} device: {} CROSS VALIDATION TRAIN END
        Elapsed time: {}
        ==============================================================
        '''.format(self.__class__.__name__, self.device, end - start))

    def run_train_cross_validation(self, k_folds_number=10):
        logging.info('''
        ==============================================================
        {} device: {} CROSS VALIDATION TRAIN START
        --------------------------------------------------------------
        '''.format(self.__class__.__name__, self.device))
        start = timer()
        for train_mask, test_mask in KFold(k_folds_number).split(self.dataset):
            train_loader = torch.utils.data.DataLoader(self.dataset,
                                                       sampler=MaskedSampler(mask=train_mask),
                                                       batch_size=self.batch_size_test,
                                                       shuffle=False)

            test_loader = torch.utils.data.DataLoader(self.dataset,
                                                      sampler=MaskedSampler(test_mask),
                                                      batch_size=self.batch_size_test,
                                                      shuffle=False)
            self.train_loop(train_loader, test_loader)
        end = timer()
        logging.info('''
        --------------------------------------------------------------
        {} device: {} CROSS VALIDATION TRAIN END
        Elapsed time: {}
        ==============================================================
        '''.format(self.__class__.__name__, self.device, end - start))

    def train(self, epoch, data_loader):
        logging.info("Epoch: {}\n".format(epoch))
        self.network.train()
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

    def test(self, data_loader):
        self.network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.network(data)
                test_loss += F.cross_entropy(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(data_loader.sampler)
        logging.info('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(data_loader.sampler),
            100. * correct / len(data_loader.sampler)))

    def train_loop(self, train_loader, test_loader):
        self.test(data_loader=test_loader)
        for epoch in range(1, self.n_epochs + 1):
            self.train(epoch=epoch, data_loader=train_loader)
            self.test(data_loader=test_loader)


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
