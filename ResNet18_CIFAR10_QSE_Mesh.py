import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import pennylane as qml

import random

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np


random_seed = 30
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB')
    print(f'Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB')

# 超参数设置
num_epochs = 100
batch_size = 128
learning_rate = 0.005

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # 因为 CIFAR10 是彩色图，有三个通道
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




import pennylane as qml


class SEBlock(nn.Module):
    def __init__(self, channel, reduction):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class QSEBlock(nn.Module):
    def __init__(self, channel, reduction):
        super(QSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            QuantumLinearLayer(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            #QuantumLinearLayer(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).float()

        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)



from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize

import torch
import numpy as np
def modeleva(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    all_labels_bin = label_binarize(all_labels, classes=[i for i in range(10)])
    all_preds_bin = label_binarize(all_preds, classes=[i for i in range(10)])

    roc_auc = roc_auc_score(all_labels_bin, all_preds_bin, average='macro', multi_class='ovo')
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    return test_loss / len(test_loader), accuracy, roc_auc, f1, precision, recall




class QuantumLinearLayer(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(QuantumLinearLayer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.quantum_input_channels = input_channels


        self.num_qubits = int(np.ceil(np.log2(self.quantum_input_channels)))
        self.dev = qml.device('default.qubit', wires=self.num_qubits)
        self.qnode = qml.QNode(self.quantum_circuit, self.dev, interface='torch')

    def amplitude_damping_kraus(self, qubit, gamma):
        r = random.random()
        if r < gamma:

            qml.RY(2 * np.arcsin(np.sqrt(gamma)), wires=qubit)
            qml.PauliX(wires=qubit)
        else:

            qml.RZ(2 * np.arccos(np.sqrt(1 - gamma)), wires=qubit)

    def phase_flipping_noise(self, qubit, gamma):
        r = random.random()
        if r < gamma:
            qml.PauliZ(wires=qubit)

    def bit_flipping_noise(self, qubit, gamma):
        r = random.random()
        if r < gamma:
            qml.PauliX(wires=qubit)

    def depolarizing_noise(self, qubit, delta):
        r = random.random()
        if r < delta / 3:
            qml.PauliX(wires=qubit)
        elif r < 2 * delta / 3:
            qml.PauliY(wires=qubit)
        elif r < delta:
            qml.PauliZ(wires=qubit)

    def quantum_circuit(self, x):
        qml.templates.embeddings.AmplitudeEmbedding(features=x, wires=range(self.num_qubits), normalize=True,
                                                    pad_with=0.0)
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                qml.CNOT(wires=[i, j])
                qml.CNOT(wires=[j, i])

        return [qml.expval(qml.PauliZ(i % self.num_qubits)) for i in range(self.output_channels)]

    def forward(self, x):
        pad_size = max(2 ** self.num_qubits - self.quantum_input_channels, 0)

        batch_outputs = []
        for sample in x:  # iterate over the batch
            padded = torch.nn.functional.pad(sample, (0, pad_size), "constant", 0)
            batch_outputs.append(self.qnode(padded))

        quantum_output = torch.stack(batch_outputs).float().to(x.device)
        return quantum_output.view(-1, self.output_channels)


import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicResidualSEBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride, r=64):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 3, padding=1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)

        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)

class BasicResidualQSEBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride, r=64):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 3, padding=1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            QuantumLinearLayer(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)

        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)

class BottleneckResidualSEBlock(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride, r=8):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):

        shortcut = self.shortcut(x)

        residual = self.residual(x)
        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)



class BottleneckResidualQSEBlock(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            QuantumLinearLayer(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):

        shortcut = self.shortcut(x)

        residual = self.residual(x)
        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)


class SEResNet(nn.Module):

    def __init__(self, block, block_num, class_num=10):
        super().__init__()

        self.in_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(block, block_num[0], 64, 1)
        self.stage2 = self._make_stage(block, block_num[1], 128, 2)
        self.stage3 = self._make_stage(block, block_num[2], 256, 2)
        self.stage4 = self._make_stage(block, block_num[3], 512, 2)

        self.linear = nn.Linear(self.in_channels, class_num)

    def forward(self, x):
        x = self.pre(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        return x


    def _make_stage(self, block, num, out_channels, stride):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        while num - 1:
            layers.append(block(self.in_channels, out_channels, 1))
            num -= 1

        return nn.Sequential(*layers)

def seresnet18():
    return SEResNet(BasicResidualSEBlock, [2, 2, 2, 2])


def qseresnet18():
    return SEResNet(BasicResidualQSEBlock, [2, 2, 2, 2])


def seresnet34():
    return SEResNet(BasicResidualSEBlock, [3, 4, 6, 3])

def seresnet50():
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 6, 3])

def qseresnet50():
    return SEResNet(BottleneckResidualQSEBlock, [3, 4, 6, 3])


def seresnet101():
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 23, 3])

def seresnet152():
    return SEResNet(BottleneckResidualSEBlock, [3, 8, 36, 3])


net_A =qseresnet18().to(device)


criterion = nn.CrossEntropyLoss()


ACNN_writer = SummaryWriter('QSEResNet18_Mesh')



Aoptimizer = optim.SGD(net_A.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0007)#0.0005

Ascheduler = optim.lr_scheduler.StepLR(Aoptimizer, step_size=30, gamma=0.01)

net_epoch_loss = []

A_net_epoch_loss = []


for epoch in range(num_epochs):

    A_net_running_loss = 0.0
    correct = 0
    total = 0
    correct1 = 0
    total1 = 0
    correct2 = 0
    total2 = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        Aoptimizer.zero_grad()
        outputs = net_A(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        Aoptimizer.step()
        A_net_running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total2 += labels.size(0)
        correct2 += (predicted == labels).sum().item()

    epoch_accuracyqse = 100 * correct2 / total2
    error_rate = (100 - epoch_accuracyqse)
    A_net_epoch_loss.append(A_net_running_loss / len(train_loader))
    Ascheduler.step()
    ACNN_writer.add_scalar('Loss/train', A_net_running_loss / len(train_loader), epoch)
    ACNN_writer.add_scalar('Accuracy/Train', epoch_accuracyqse, epoch)
    ACNN_writer.add_scalar('ErrorRate/Train', error_rate, epoch)
    print(f'Qse Mesh: Epoch {epoch + 1}, Loss: {A_net_running_loss / len(train_loader)}')



    test_loss, accuracy, roc_auc, f1, precision, recall = modeleva(net_A, test_loader, criterion)
    error_rate = (100 - accuracy)

    ACNN_writer.add_scalar('ErrorRate/Test', error_rate, epoch)
    ACNN_writer.add_scalar('Accuracy/Test', accuracy, epoch)
    ACNN_writer.add_scalar('ROC AUC', roc_auc, epoch)
    ACNN_writer.add_scalar('F1 Score', f1, epoch)
    ACNN_writer.add_scalar('Precision', precision, epoch)
    ACNN_writer.add_scalar('Recall', recall, epoch)

ACNN_writer.close()
