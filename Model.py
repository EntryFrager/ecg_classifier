import torch
import torch.nn as nn
import torch.nn.functional as F


device = "cuda" if torch.cude.is_available() else "cpu"


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, planes, width, stride=1, downsample = None):
        super().__init__()

        self.conv_1 = nn.Conv1d(planes, width, kernel_size=1, stride=1, padding=0),
        self.batch_norm_1 = nn.BatchNorm1d(width)
        self.conv_2 = nn.Conv1d(width, width, kernel_size=3, stride=stride, padding=1),
        self.batch_norm_2 = nn.BatchNorm1d(width)
        self.conv_3 = nn.Conv1d(width, width * self.expansion, kernel_size=1, stride=1, padding=0),
        self.batch_norm_3 = nn.BatchNorm1d(width * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    
    def forward(self, x):
        identity = x

        out = self.relu(self.batch_norm_1(self.conv_1(x)))
        out = self.relu(self.batch_norm_2(self.conv_2(out)))
        out = self.batch_norm_3(self.conv_3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classess):
        super().__init__()

        self.inplanes = 64
        self.dilation = 1
        bottle_neck = Bottleneck()
        
        self.conv_1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False),
        self.batch_norm_1 = nn.BatchNorm1d(64),
        self.relu = nn.ReLU(),
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer_1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer_2 = self._make_layer(Bottleneck, 128, layers[0])
        self.layer_3 = self._make_layer(Bottleneck, 216, layers[0])
        self.layer_4 = self._make_layer(Bottleneck, 512, layers[0])

        self.avg_pool = nn.AvgPool1d(kernel_size=1)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classess)


    def forward(self, x):
        out = self.conv_1(x)
        out = self.batch_norm_1(out)
        out = self.relu(out)
        out = self.maxpool_1(out)

        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)

        out = self.avg_pool(out)
        out = self.fc(out)
        return out


    def _make_layer(self, ):

        return


def train(net, train_dataset, val_dataset, n_epoch, optimizer, criterion):
    return


def test(net, test_dataset, criterion):
    return
