import torch
import torch.nn as nn
import torch.nn.functional as F


device = "cuda" if torch.cude.is_available() else "cpu"


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample = None):
        super().__init__()

        self.conv_1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, padding=0),
        self.batch_norm_1 = nn.BatchNorm1d(planes)
        self.conv_2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1),
        self.batch_norm_2 = nn.BatchNorm1d(planes)
        self.conv_3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0),
        self.batch_norm_3 = nn.BatchNorm1d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    
    def forward(self, x):
        residual = x

        out = self.conv_1(x)
        out = self.batch_norm_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.batch_norm_2(out)
        out = self.relu(out)

        out = self.conv_3(out)
        out = self.batch_norm_3(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classess=1000):
        super().__init__()

        self.inplanes = 64

        self.conv_1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm_1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer_1 = self._make_layer(block, 64, layers[0])
        self.layer_2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer_3 = self._make_layer(block, 216, layers[2], stride=2)
        self.layer_4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avg_pool = nn.AvgPool1d(kernel_size=1)
        self.fc = nn.Linear(512 * block.expansion, num_classess)


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
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes ** block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


def train(net, train_loader, val_loader, 
          n_epoch, optimizer, criterion):
    loss_train_history = []
    loss_val_history = []

    for epoch in range(n_epoch):
        print('Epoch {}/{}:'.format(epoch + 1, n_epoch), flush = True)

        train_loss = train_acc = val_acc = val_loss = 0.0

        net.train()

        for (batch_idx, train_batch) in enumerate(train_loader):
            features, labels = train_batch[0].to(device), train_batch[1].to(device)
            optimizer.zero_grad()

            preds = net(features)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc  += (preds.argmax(dim = 1) == labels).sum().item()

        train_acc  /= len(train_loader.dataset)
        train_loss /= len(train_loader)
        loss_train_history.append(train_loss)

        net.eval()

        with torch.no_grad():
            for val_batch in val_loader:
                images, labels = val_batch[0].to(device), val_batch[1].to(device)
                preds = net(images)
                val_acc  += (preds.argmax(axis = 1) == labels).sum().item()
                val_loss += criterion(preds, labels).item()

        val_acc  /= len(val_loader.dataset)
        val_loss /= len(val_loader)
        loss_val_history.append(val_loss)

        print(f'train Loss: {train_loss:.4f} Acc: {train_acc:.4f}\n'
              f'val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    
    return net, loss_train_history, loss_val_history


def test(net, test_loader, criterion):
    net.eval()

    test_acc = test_loss = 0.0

    with torch.no_grad():
        for (batch_idx, test_batch) in enumerate(test_loader): 
            images, labels = test_batch[0].to(device), test_batch[1].to(device)
            preds = net(images)

            test_acc  += (preds.argmax(axis = 1) == labels).sum().item()
            test_loss += criterion(preds, labels).item()

    test_acc  /= len(test_loader.dataset)
    test_loss /= len(test_loader)

    return test_loss, test_acc
