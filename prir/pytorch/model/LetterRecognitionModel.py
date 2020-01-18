import torch
import pandas as pd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from prir.path.utils import get_datasets_path
from prir.pytorch.model.NetworkModel import NetworkModel, weight_init


class LetterRecognitionModel(NetworkModel):
    def __init__(self,
                 device='cuda:0',
                 train_to_test_split_ratio=0.3,
                 batch_size=128,
                 n_epochs=300):
        super().__init__( device=device, train_to_test_split_ratio=train_to_test_split_ratio,
                       batch_size=batch_size, n_epochs=n_epochs)

    def __init_class_fields__(self):
        self.num_classes = 26
        self.learning_rate = 0.001
        self.dataset = LetterRecognitionDataset(csv_file=get_datasets_path()
                                                .joinpath('letter-recognition/letter-recognition.csv'))

    def __init_network__(self):
        self.network = Net(self.batch_size_train)
        self.network.apply(weight_init)
        self.network.to(self.device)
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=0.3,
                                   weight_decay=1e-7)


class LetterRecognitionDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data = pd.read_csv(csv_file, header=None)
        self.y_data = torch.tensor(pd.factorize(data[0])[0])
        self.x_data = torch.tensor(data.drop(data.columns[0], axis=1).values.astype('float32'))
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x_data[idx], self.y_data[idx]

class Net(nn.Module):
    def __init__(self, output_size):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(16, 50)
        self.dense2 = nn.Linear(50, 22)
        self.dense3 = nn.Linear(22, output_size)

    def forward(self, x):
        x = F.elu(self.dense1(x))
        x = F.dropout(x, training=self.training, p=0.5)
        x = F.elu(self.dense2(x))
        x = F.dropout(x, training=self.training, p=0.5)
        return F.elu(self.dense3(x))