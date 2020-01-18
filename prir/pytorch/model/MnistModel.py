import torch.nn as nn
import torchvision
from torch import optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from prir.path.utils import get_datasets_path
from prir.pytorch.model.NetworkModel import NetworkModel, weight_init


class MnistModel(NetworkModel):
    def __init_class_fields__(self):
        self.num_classes = 10
        self.learning_rate = 1.0
        self.dataset = ConcatDataset([
            torchvision.datasets.MNIST(get_datasets_path().joinpath('mnist'), train=True, download=True,
                                       transform=torchvision.transforms.ToTensor()),
            torchvision.datasets.MNIST(get_datasets_path().joinpath('mnist'), train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
        ])

    def __init_network__(self):
        self.network = Net(self.batch_size_train)
        self.network.apply(weight_init)
        self.network.to(self._device)
        self.optimizer = optim.Adadelta(self.network.parameters(), lr=self.learning_rate, rho=0.95)


class Net(nn.Module):
    def __init__(self, output_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dense1 = nn.Linear(64 * 12 * 12, 128)
        self.dense2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool1(x)
        x = F.dropout(x, training=self.training, p=0.25)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, training=self.training, p=0.5)
        # no need to use softmax because CrossEntropyLoss does that
        # x = F.softmax(self.dense2(x))
        return self.dense2(x)
