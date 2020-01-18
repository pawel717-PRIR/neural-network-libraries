import torch.nn as nn
import torchvision
from torch import optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from prir.path.utils import get_datasets_path
from prir.pytorch.model.NetworkModel import NetworkModel, weight_init


class Cifar10Model(NetworkModel):
    def __init_class_fields__(self):
        self.num_classes = 10
        self.learning_rate = 0.0001
        self.dataset = ConcatDataset([
            torchvision.datasets.CIFAR10(get_datasets_path().joinpath('cifar10'), train=True, download=True,
                                       transform=torchvision.transforms.ToTensor()),
            torchvision.datasets.CIFAR10(get_datasets_path().joinpath('cifar10'), train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
        ])

    def __init_network__(self):
        self.network = Net(self.batch_size_train)
        self.network.apply(weight_init)
        self.network.to(self._device)
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate, eps=1e-7)


class Net(nn.Module):
    def __init__(self, output_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dense1 = nn.Linear(64 * 5 * 5, 512)
        self.dense2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool1(x)
        x = F.dropout(x, training=self.training, p=0.25)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.max_pool2(x)
        x = F.dropout(x, training=self.training, p=0.25)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, training=self.training, p=0.5)
        # no need to use softmax because CrossEntropyLoss does that
        return self.dense2(x)
