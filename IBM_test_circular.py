import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import torch
from torch.utils.data import Subset, DataLoader, random_split

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
import torchvision.transforms as T

num_epochs = 100
batch_size = 1
learning_rate = 0.005
num_train    = 20
num_test     = 10

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010))
])


full_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
full_test  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)


small_train, _ = random_split(full_train,
                              [num_train, len(full_train)-num_train],
                              generator=torch.Generator().manual_seed(42))

small_test,  _ = random_split(full_test,
                              [num_test, len(full_test)-num_test],
                              generator=torch.Generator().manual_seed(42))


train_loader = DataLoader(small_train, batch_size=batch_size,
                          shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(small_test,  batch_size=batch_size,
                          shuffle=False, num_workers=0, pin_memory=True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(channel="ibm_quantum",
                                  token="",
                                  overwrite=True, set_as_default=True)
service = QiskitRuntimeService(channel="ibm_quantum",
                               token="")

import torch
import torch.nn as nn


from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session


class QuantumLinearLayer(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, num_qubits: int|None=None, shots: int=1024):
        super().__init__()

        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.num_qubits = (max(2, int(np.ceil(np.log2(input_dim))))
                           if num_qubits is None else max(1, num_qubits))
        self.shots = shots


        self.theta        = nn.Parameter(torch.rand(self.num_qubits))
        self.theta_qiskit = ParameterVector("θ", length=self.num_qubits)
        self.x_qiskit     = ParameterVector("x", length=self.num_qubits)


        qc = QuantumCircuit(self.num_qubits)
        for q in range(self.num_qubits):
            qc.rx(self.x_qiskit[q], q)
        for q in range(self.num_qubits):
            qc.ry(self.theta_qiskit[q], q)
        for q in range(self.num_qubits - 1):
            qc.cx(q, q + 1)
        if self.num_qubits > 2:
            qc.cx(self.num_qubits - 1, 0)


        qc.measure_all()


        self.circuit = qc


        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=self.num_qubits)
        self.backend = backend
        self.session = Session(backend=backend)
        self.sampler = Sampler(mode=self.session)

    def _counts_to_expectation(self, counts: dict[str, int]) -> torch.Tensor:
        total = sum(counts.values())
        expect = torch.zeros(self.num_qubits)
        for bitstring, cnt in counts.items():
            bits = bitstring[::-1]
            for q in range(self.num_qubits):

                expect[q] += cnt * (1.0 if bits[q] == "0" else -1.0)
        return expect / total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for sample in x:

            bind_params = {}
            for q in range(self.num_qubits):
                bind_params[self.x_qiskit[q]]     = sample[q % self.input_dim].item()
                bind_params[self.theta_qiskit[q]] = self.theta[q].item()


            circuit_bound = self.circuit.assign_parameters(bind_params)


            circuit_transpiled = transpile(circuit_bound, backend=self.backend, optimization_level=0)


            job = self.sampler.run(
                [circuit_transpiled]
            )

            res = job.result()
            counts = res[0].data.meas.get_counts()


            z_exp = self._counts_to_expectation(counts)
            repeat_factor = int(np.ceil(self.output_dim / self.num_qubits))
            outputs.append(z_exp.repeat(repeat_factor)[:self.output_dim])

        return torch.stack(outputs).to(x.dtype)

    def __del__(self):
        try:
            self.session.close()
        except Exception:
            pass



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


    return test_loss / len(test_loader), accuracy

import torch.nn as nn
import torch.nn.functional as F


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

def qseresnet18():
    return SEResNet(BasicResidualQSEBlock, [2, 2, 2, 2])





net_A =qseresnet18().to(device)


criterion = nn.CrossEntropyLoss()


ACNN_writer = SummaryWriter('IBM/QSEResNet18_Circular')

Aoptimizer = optim.SGD(net_A.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0007)#0.0005

Ascheduler = optim.lr_scheduler.StepLR(Aoptimizer, step_size=30, gamma=0.01)


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
    print(f'Qse IBM Circular: Epoch {epoch + 1}, Loss: {A_net_running_loss / len(train_loader)}')



    test_loss, accuracy= modeleva(net_A, test_loader, criterion)
    error_rate = (100 - accuracy)

    ACNN_writer.add_scalar('ErrorRate/Test', error_rate, epoch)
    ACNN_writer.add_scalar('Accuracy/Test', accuracy, epoch)
    # ACNN_writer.add_scalar('ROC AUC', roc_auc, epoch)
    # ACNN_writer.add_scalar('F1 Score', f1, epoch)
    # ACNN_writer.add_scalar('Precision', precision, epoch)
    # ACNN_writer.add_scalar('Recall', recall, epoch)

ACNN_writer.close()
