import torch
import torch.nn as nn
import numpy as np
import os
import warnings
import random
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

warnings.filterwarnings("ignore")
DEFAULT_RANDOM_SEED = 42


def SeedBasic(seed = DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def SeedTorch(seed = DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def SeedEverything(seed = DEFAULT_RANDOM_SEED):
    SeedBasic(seed)
    SeedTorch(seed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training will take on {device}")

SeedEverything()


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample = None):
        super().__init__()

        self.conv_1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, padding=0)
        self.batch_norm_1 = nn.BatchNorm1d(planes)
        self.conv_2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.batch_norm_2 = nn.BatchNorm1d(planes)
        self.conv_3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm_3 = nn.BatchNorm1d(planes * self.expansion)

        self.relu = nn.ReLU()
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

        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()

        self.inplanes = 64

        self.conv_1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.batch_norm_1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer_1 = self._make_layer(block, 64, layers[0])
        self.layer_2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer_3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer_4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


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
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


def train(net, train_loader, val_loader, 
          n_epoch, optimizer, criterion, treshold_preds):
    train_history = []
    val_history   = []

    for epoch in range(n_epoch):
        print('Epoch {}/{}:'.format(epoch + 1, n_epoch), flush = True)

        train_loss = val_loss = 0.0
        train_labels, train_preds, train_prob, val_labels, val_preds, val_prob = [], [], [], [], [], []

        net.train()

        for (batch_idx, train_batch) in enumerate(train_loader):
            samples, labels = train_batch['ecg_signals'].to(device), train_batch['labels'].to(device)
            optimizer.zero_grad()

            preds = net(samples)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                preds = torch.sigmoid(preds)
                bin_preds = (preds > treshold_preds).float()
                
                train_prob.append(preds.cpu().numpy())
                train_labels.append(labels.cpu().numpy())
                train_preds.append(bin_preds.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_metrics = metric_func(np.concatenate(train_labels), np.concatenate(train_preds), np.concatenate(train_prob))
        train_metrics['loss'] = train_loss
        train_history.append(train_metrics)

        net.eval()

        with torch.no_grad():
            for val_batch in val_loader:
                samples, labels = val_batch['ecg_signals'].to(device), val_batch['labels'].to(device)
                preds = net(samples)
                val_loss += criterion(preds, labels).item()

                preds = torch.sigmoid(preds)

                bin_preds = (preds > treshold_preds).float()

                val_prob.append(preds.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
                val_preds.append(bin_preds.cpu().numpy())

        val_loss /= len(val_loader)
        val_metrics = metric_func(np.concatenate(val_labels), np.concatenate(val_preds), np.concatenate(val_prob))
        val_metrics['loss'] = val_loss
        val_history.append(val_metrics)
        
        print(f'Train metrics: \n\tf1 mean score = {train_metrics['f1_mean']:.4f}, \n\troc auc mean = {train_metrics['roc_auc_mean']:.4f}, \
              \n\tsensitivity = {train_metrics['sens_mean']:.4f}, \n\tspecifisity = {train_metrics['spec_mean']:.4f}, \n\tloss = {train_metrics['loss']:.4f}\n'
              f'Validation metrics: \n\tf1 mean score = {val_metrics['f1_mean']:.4f}, \n\troc auc mean = {val_metrics['roc_auc_mean']:.4f}, \
              \n\tsensitivity = {val_metrics['sens_mean']:.4f}, \n\tspecifisity = {val_metrics['spec_mean']:.4f}, \n\tloss = {val_metrics['loss']:.4f}\n\n')
    
    return net, train_history, val_history


def test(net, test_loader, criterion, treshold_preds):
    net.eval()

    test_loss = 0.0
    test_preds, test_labels, test_prob = [], [], []

    with torch.no_grad():
        for (batch_idx, test_batch) in enumerate(test_loader): 
            samples, labels = test_batch['ecg_signals'].to(device), test_batch['labels'].to(device)
            preds = net(samples)

            test_loss += criterion(preds, labels).item()

            preds = torch.sigmoid(preds)

            bin_preds = (preds > treshold_preds).float()
            test_prob.append(preds.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            test_preds.append(bin_preds.cpu().numpy())

    test_loss /= len(test_loader)
    test_metrics = metric_func(np.concatenate(test_labels), np.concatenate(test_preds), np.concatenate(test_prob))
    test_metrics['loss'] = test_loss

    print(f'Test metrics: \n\tf1 mean score = {test_metrics['f1_mean']:.4f}, \n\troc auc mean = {test_metrics['roc_auc_mean']:.4f}, \
          \n\tsensitivity = {test_metrics['sens_mean']:.4f}, \n\tspecifisity = {test_metrics['spec_mean']:.4f}, \n\tloss = {test_metrics['loss']:.4f}\n')

    return test_metrics

def metric_func(bin_labels, bin_preds, preds):
    f1_acc      = []
    sensitivity = []
    specificity = []
    roc_auc     = []

    for i in range(0, bin_labels.shape[1]):
        conf_matrix = confusion_matrix(bin_labels[:, i], bin_preds[:, i], labels=[0, 1])
        sensitivity.append(conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0]) if (conf_matrix[1][1] + conf_matrix[1][0]) != 0 else 0)
        specificity.append(conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1]) if (conf_matrix[0][0] + conf_matrix[0][1]) != 0 else 0)
        f1_acc.append(f1_score(bin_labels[:, i], bin_preds[:, i], zero_division=0))
        roc_auc.append(roc_auc_score(bin_labels[:, i], preds[:, i]))

    return {
        'f1_acc': f1_acc,
        'f1_mean': np.mean(f1_acc),
        'sens': sensitivity,
        'sens_mean': np.mean(sensitivity),
        'spec': specificity,
        'spec_mean': np.mean(specificity),
        'roc_auc': roc_auc,
        'roc_auc_mean': np.mean(roc_auc),
        'confusion_matrix': conf_matrix
    }

