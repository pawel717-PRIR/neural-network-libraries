import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


def get_device(use_gpu=False):
    dev = "cpu"
    if (use_gpu):
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            print("WARING: GPU not available. Fallback to CPU")
    return torch.device(dev)


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dense1 = nn.Linear(64 * 5 * 5, 512)
        self.dense2 = nn.Linear(512, batch_size_train)

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


def train(epoch):
    print("Epoch: {}\n".format(epoch))
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


num_classes = 10
n_epochs = 12
batch_size_train = 128
batch_size_test = batch_size_train
learning_rate = 0.0001
log_interval = batch_size_train

device = get_device(use_gpu=True)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('CIFAR_data/', train=True, download=True,
                               transform=torchvision.transforms.ToTensor()),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('CIFAR_data/', train=False, download=True,
                               transform=torchvision.transforms.ToTensor()),
    batch_size=batch_size_test, shuffle=True)

network = Net()
network.apply(weight_init)
network.to(device)

optimizer = optim.RMSprop(network.parameters(), lr=learning_rate, eps=1e-7)
train_counter = []

test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
