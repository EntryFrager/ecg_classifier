import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import os
import warnings
import random
import copy
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_curve

warnings.filterwarnings("ignore")
DEFAULT_RANDOM_SEED = 42


def SeedBasic(seed = DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def SeedTorch(seed = DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def SeedEverything(seed = DEFAULT_RANDOM_SEED):
    SeedBasic(seed)
    SeedTorch(seed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training will take on {device}")

SeedEverything()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv_1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm1d(planes)

        self.downsample = downsample
        self.stride = stride

    
    def forward(self, x):
        residual = x

        out = self.conv_1(x)
        out = self.batch_norm_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.batch_norm_2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample = None):
        super().__init__()

        self.conv_1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm_1 = nn.BatchNorm1d(planes)
        self.conv_2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm1d(planes)
        self.conv_3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
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

        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()

        self.inplanes = 64

        self.conv_1 = nn.Conv1d(12, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)
        self.batch_norm_1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
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
    

class EarlyStopping:
    def __init__(self, patience, loss_delta, acc_delta):
        self.best_loss  = None
        self.best_sens  = None
        self.best_spec  = None
        self.best_model = None
        self.best_threshold = None

        self.loss_delta = loss_delta
        self.acc_delta  = acc_delta

        self.patience = patience
        self.counter  = 0

        self.stop = False


    def __call__(self, loss, sens, spec, net, threshold):
        if self.best_loss is None and self.best_sens is None and self.best_spec is None:
                self.best_loss  = loss
                self.best_sens  = sens
                self.best_spec  = spec
                self.best_model = copy.deepcopy(net)
                self.best_threshold = threshold
        elif loss >= self.best_loss - self.loss_delta and (self.best_sens - self.acc_delta <= sens <= self.best_sens + self.acc_delta or spec <= self.best_spec + self.acc_delta):
            if loss <= self.best_loss and (sens >= self.best_sens or spec >= self.best_spec):
                self.best_loss  = loss
                self.best_sens  = sens
                self.best_spec  = spec
                self.best_model = copy.deepcopy(net)
                self.best_threshold = threshold
                self.counter = 0
                print(f'Best Loss: {self.best_loss:.4f}\n'
                      f'Best Sens: {self.best_sens:.4f}\n'
                      f'Best Spec: {self.best_spec:.4f}')
            else:
                self.counter += 1
        else:
            self.counter += 1

        print(f"EarlyStopping: {self.counter} / {self.patience}")

        if self.counter >= self.patience:
            self.stop = True

        return self.stop

    
    def get_best_net(self):
        return self.best_model
    

    def get_best_threshold(self):
        return self.best_threshold


def train(net, train_loader, val_loader, 
          n_epoch, optimizer, criterion, 
          threshold_preds, patience, loss_delta, acc_delta):
    loss_train_history = []
    loss_val_history   = []

    writer = SummaryWriter(log_dir='logs')
    early_stop = EarlyStopping(patience, loss_delta, acc_delta)

    for epoch in range(n_epoch):
        print('Epoch {}/{}:'.format(epoch + 1, n_epoch), flush = True)

        train_loss = val_loss = 0.0
        val_labels, val_prob = [], []

        net.train()

        for (batch_idx, train_batch) in enumerate(train_loader):
            samples, labels = train_batch['ecg_signals'].to(device), train_batch['labels'].to(device)
            optimizer.zero_grad()

            preds = net(samples)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        loss_train_history.append(train_loss)

        net.eval()

        with torch.no_grad():
            for val_batch in val_loader:
                samples, labels = val_batch['ecg_signals'].to(device), val_batch['labels'].to(device)
                preds = net(samples)
                val_loss += criterion(preds, labels).item()

                preds = torch.sigmoid(preds)

                val_prob.append(preds.cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_loss /= len(val_loader)
        loss_val_history.append(val_loss)

        print('\nValidation metrics:')
        val_sens, val_spec, threshold_preds = get_metrics(np.concatenate(val_labels), np.concatenate(val_prob), test=False)
        
        print(f'\ntrain Loss: {train_loss:.4f}'
              f'\nval Loss: {val_loss:.4f}')
        
        writer.add_scalars('Loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)

        if early_stop(val_loss, val_sens, val_spec, net, threshold_preds):
            break
    
    writer.close()
    
    torch.save(early_stop.get_best_net().state_dict(), 'save_best_models/best_model.pt')
    torch.save(early_stop.get_best_threshold(), 'save_best_models/best_threshold.pt')
    
    return early_stop.get_best_net(), early_stop.get_best_threshold(), loss_train_history, loss_val_history


def test(net, test_loader, criterion, treshold_preds):
    net.eval()

    test_loss = 0.0
    test_labels, test_prob = [], []

    with torch.no_grad():
        for (batch_idx, test_batch) in enumerate(test_loader): 
            samples, labels = test_batch['ecg_signals'].to(device), test_batch['labels'].to(device)
            preds = net(samples)

            test_loss += criterion(preds, labels).item()

            preds = torch.sigmoid(preds)

            test_prob.append(preds.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            
    test_loss /= len(test_loader)

    print('\nTest metrics:')
    get_metrics(np.concatenate(test_labels), np.concatenate(test_prob), test=True, threshold=treshold_preds)

    print(f'\ntest Loss: {test_loss:.4f}')

    return test_loss


def get_metrics(y_true, y_probs, test=False, threshold=None):
    TP, FP, TN, FN = [], [], [], []
    sensitivity, specificity, precision = [], [], []
    roc_auc = []

    best_threshold = []

    if test:
        y_preds = (y_probs >= threshold).astype(np.float32)

    print('Confusion matrix:')
    for i in range(0, y_true.shape[1]):
        if test:
            conf_matrix = confusion_matrix(y_true[:, i], y_preds[:, i], labels=[0, 1])
            tn, fp, fn, tp = conf_matrix.ravel()

            print(pd.DataFrame([{'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}]).to_string(index=False))
            
            sensitivity_i = tp / (tp + fn)
            specificity_i = tn / (tn + fp)
            precision_i   = tp / (tp + fp)

            TP.append(tp)
            FP.append(fp)
            TN.append(tn)
            FN.append(fn)

            sensitivity.append(sensitivity_i)
            specificity.append(specificity_i)
            precision.append(precision_i)
        else:
            thresholds = np.unique(y_probs[:, i])

            TP_i, FP_i, TN_i, FN_i = [], [], [], []
            sensitivity_i, specificity_i, precision_i = [], [], []
            f1_i = []

            for threshold in thresholds:
                preds = (y_probs[:, i] >= threshold).astype(np.float32)
                cm = confusion_matrix(y_true[:, i], preds, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn)
                spec = tn / (tn + fp)
                prec = tp / (tp + fp)
                f1   = 2 * sens * prec / (sens + prec) if sens + prec != 0 else 0

                TP_i.append(tp)
                FP_i.append(fp)
                TN_i.append(tn)
                FN_i.append(fn)

                sensitivity_i.append(sens)
                specificity_i.append(spec)
                precision_i.append(prec)

                f1_i.append(f1)

            best_idx = np.argmax(f1_i)
            best_threshold.append(thresholds[best_idx])

            print(pd.DataFrame([{'TP': TP_i[best_idx], 'FP': FP_i[best_idx], 'TN': TN_i[best_idx], 'FN': FN_i[best_idx]}]).to_string(index=False))

            TP.append(TP_i[best_idx])
            FP.append(FP_i[best_idx])
            TN.append(TN_i[best_idx])
            FN.append(FN_i[best_idx])

            sensitivity.append(sensitivity_i[best_idx])
            specificity.append(specificity_i[best_idx])
            precision.append(precision_i[best_idx])

        roc_auc.append(roc_auc_score(y_true[:, i], y_probs[:, i]))
        
    # micro averaging

    TP_mean, FP_mean, TN_mean, FN_mean = np.mean(TP), np.mean(FP), np.mean(TN), np.mean(FN)
    micro_sensitivity = TP_mean / (TP_mean + FN_mean)
    micro_specificity = TN_mean / (TN_mean + FP_mean)
    micro_precision   = TP_mean / (TP_mean + FP_mean)

    micro_f1 = 2 * micro_sensitivity * micro_precision / (micro_sensitivity + micro_precision)

    print('\nMicro averaging:')
    print(pd.DataFrame.from_dict({
        'sensitivity': micro_sensitivity,
        'specificity': micro_specificity,
        'precision':   micro_precision,
        'f1 score':    micro_f1
    }, orient='index').to_string(header=False))

    # macro averaging

    macro_sensitivity = np.mean(sensitivity)
    macro_specificity = np.mean(specificity)
    macro_precision   = np.mean(precision)

    macro_f1 = 2 * macro_sensitivity * macro_precision / (macro_sensitivity + macro_precision)

    print('\nMacro averaging:')
    print(pd.DataFrame.from_dict({
        'sensitivity': macro_sensitivity,
        'specificity': macro_specificity,
        'precision':   macro_precision,
        'f1 score':    macro_f1
    }, orient='index').to_string(header=False))

    # roc auc and classification report

    print(f'\nROC AUC: {np.mean(roc_auc):.4f}')

    if not test:
        y_preds = (y_probs >= best_threshold).astype(np.float32)

    print(f'\nClassification report from sklearn:\n{classification_report(y_true, y_preds)}')

    return macro_sensitivity, macro_specificity, best_threshold
