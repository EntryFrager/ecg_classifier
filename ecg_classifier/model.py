import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import os
import warnings
import random
import copy
import hydra
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_curve
from typing import Optional, Tuple, List, Type, Any


warnings.filterwarnings("ignore")
DEFAULT_RANDOM_SEED = 42


def SeedBasic(seed: int = DEFAULT_RANDOM_SEED) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def SeedTorch(seed: int = DEFAULT_RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def SeedEverything(seed: int = DEFAULT_RANDOM_SEED) -> None:
    SeedBasic(seed)
    SeedTorch(seed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training will take on {device}")

SeedEverything()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, 
                 inplanes: int, 
                 planes: int, 
                 stride: int = 1, 
                 drop_prob: float = 0.0, 
                 downsample: Optional[nn.Module] = None) -> None:
        super().__init__()

        self.conv_1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm1d(planes)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm1d(planes)
        self.relu_2 = nn.ReLU()

        self.drop_1 = nn.Dropout1d(p=drop_prob)
        self.drop_2 = nn.Dropout1d(p=drop_prob)

        self.downsample = downsample
        self.stride = stride

    
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv_1(x)
        out = self.batch_norm_1(out)
        out = self.relu_1(out)
        out = self.drop_1(out)

        out = self.conv_2(out)
        out = self.batch_norm_2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu_2(out)
        out = self.drop_2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, 
                 inplanes: int, 
                 planes: int, 
                 stride: int = 1, 
                 drop_prob: float = 0.0, 
                 downsample: Optional[nn.Module] = None) -> None:
        super().__init__()

        self.conv_1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm_1 = nn.BatchNorm1d(planes)
        self.drop_1 = nn.Dropout1d(p=drop_prob)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm1d(planes)
        self.drop_2 = nn.Dropout1d(p=drop_prob)        
        self.relu_2 = nn.ReLU()
        self.conv_3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm_3 = nn.BatchNorm1d(planes * self.expansion)
        self.drop_3 = nn.Dropout1d(p=drop_prob)
        self.relu_3 = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv_1(x)
        out = self.batch_norm_1(out)
        out = self.relu_1(out)
        out = self.drop_1(out)

        out = self.conv_2(out)
        out = self.batch_norm_2(out)
        out = self.relu_2(out)
        out = self.drop_2(out)

        out = self.conv_3(out)
        out = self.batch_norm_3(out)
        out = self.relu_3(out)
        out = self.drop_3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out


class ResNet(nn.Module):
    def __init__(self, 
                 block: str, 
                 layers: List[int], 
                 num_classes: int = 1000, 
                 drop_prob_head: float = 0.5, 
                 drop_prob_backbone: float = 0.0):
        super().__init__()

        if isinstance(block, str):
            block = hydra.utils.get_class(block)

        self.inplanes = 64
        self.drop_prob_backbone = drop_prob_backbone

        self.conv_1 = nn.Conv1d(12, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer_1 = self._make_layer(block, 64, layers[0])
        self.layer_2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer_3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer_4 = self._make_layer(block, 512, layers[3], stride=2)

        self.drop = nn.Dropout1d(p=drop_prob_head)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
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
        out = self.drop(out)
        out = self.fc(out)

        return out


    def _make_layer(self, 
                    block: Type[nn.Module], 
                    planes: int, 
                    blocks: int, 
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, self.drop_prob_backbone, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, drop_prob=self.drop_prob_backbone))

        return nn.Sequential(*layers)


class EarlyStopping:
    def __init__(self, 
                 patience: int) -> None:
        self.best_loss  = None
        self.best_model = None
        self.best_threshold = None

        self.patience = patience
        self.counter  = 0

        self.stop = False


    def __call__(self, 
                 loss: float, 
                 sens: float, 
                 spec: float, 
                 net: nn.Module, 
                 threshold: np.ndarray) -> bool:
        if self.best_loss is None and self.best_sens is None and self.best_spec is None:
                self.best_loss  = loss
                self.best_sens  = sens
                self.best_spec  = spec
                self.best_model = copy.deepcopy(net)
                self.best_threshold = threshold
        elif loss <= self.best_loss:
            self.best_loss  = loss
            self.best_model = copy.deepcopy(net)
            self.best_threshold = threshold
            self.counter = 0
            print(f'\nBest Loss: {self.best_loss:.4f}\n'
                  f'Best threshold: {self.best_threshold}')
        else:
            self.counter += 1

        print(f"EarlyStopping: {self.counter} / {self.patience}\n")

        if self.counter >= self.patience:
            self.stop = True

        return self.stop


def train(net: nn.Module, 
          train_loader: torch.utils.data.DataLoader, 
          val_loader: torch.utils.data.DataLoader, 
          n_epoch: int, 
          optimizer: torch.optim.Optimizer, 
          criterion: nn.Module, 
          scheduler: Any, 
          early_stopping: EarlyStopping) -> Tuple[nn.Module, np.ndarray, List[float], List[float]]:
    loss_train_history = []
    loss_val_history   = []

    threshold_preds = []

    writer = SummaryWriter(log_dir='logs')

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

        val_labels = np.concatenate(val_labels)
        val_prob   = np.concatenate(val_prob)

        scheduler.step(val_loss)
        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']}")

        threshold_preds = find_best_threshold(val_labels, val_prob)

        print('\nValidation metrics:')
        val_sens, val_spec = get_metrics(val_labels, val_prob, threshold_preds)
        
        print(f'\ntrain Loss: {train_loss:.4f}'
              f'\nval Loss: {val_loss:.4f}')
        
        writer.add_scalars('Loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch + 1)

        if early_stopping(val_loss, val_sens, val_spec, net, threshold_preds):
            break
    
    writer.close()
    
    torch.save(early_stopping.best_model.state_dict(), 'save_best_models/best_model.pt')
    torch.save(early_stopping.best_threshold, 'save_best_models/best_threshold.pt')
    
    return early_stopping.best_model, early_stopping.best_threshold, loss_train_history, loss_val_history


def test(net: nn.Module,
         test_loader: torch.utils.data.DataLoader, 
         criterion: nn.Module, 
         threshold_preds: np.ndarray) -> float:
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
    get_metrics(np.concatenate(test_labels), np.concatenate(test_prob), threshold_preds)

    print(f'\ntest Loss: {test_loss:.4f}')

    return test_loss


def get_metrics(y_true: np.ndarray, 
                y_probs: np.ndarray,
                threshold: np.ndarray) -> Tuple[float, float]:
    tp, fp, tn, fn = [], [], [], []
    sensitivity, specificity, precision = [], [], []
    roc_auc = []

    y_pred = (y_probs >= threshold).astype(np.float32)

    print('Confusion matrix:')
    for i in range(0, y_true.shape[1]):
        tp_i, fp_i, tn_i, fn_i, sens, spec, prec = compute_confusion_metrics(y_true[:, i], y_pred[:, i])

        print(pd.DataFrame([{'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}]).to_string(index=False))

        tp.append(tp_i)
        fp.append(fp_i)
        tn.append(tn_i)
        fn.append(fn_i)

        sensitivity.append(sens)
        specificity.append(spec)
        precision.append(prec)

        roc_auc.append(roc_auc_score(y_true[:, i], y_probs[:, i]))
        
        
    # micro averaging

    micro_sens, micro_spec, micro_prec, micro_f1 = compute_micro_average(tp, fp, tn, fn)

    print('\nMicro averaging:')
    print(pd.DataFrame.from_dict({
        'sensitivity': micro_sens,
        'specificity': micro_spec,
        'precision':   micro_prec,
        'f1 score':    micro_f1
        }, orient='index').to_string(header=False))

    # macro averaging

    macro_sens, macro_spec, macro_prec, macro_f1 = compute_macro_average(sens, spec, prec)

    print('\nMacro averaging:')
    print(pd.DataFrame.from_dict({
        'sensitivity': macro_sens,
        'specificity': macro_spec,
        'precision':   macro_prec,
        'f1 score':    macro_f1
    }, orient='index').to_string(header=False))

    # roc auc and classification report

    print(f'\nROC AUC: {np.mean(roc_auc):.4f}')

    print(f'\nClassification report from sklearn:\n{classification_report(y_true, y_pred)}')

    return macro_sens, macro_spec


def compute_confusion_metrics(y_true_class: np.ndarray, 
                              y_pred_class: np.ndarray) -> Tuple[int, int, int, int, float, float, float]:
    conf_matrix = confusion_matrix(y_true_class, y_pred_class, labels=[0, 1])
    tn, fp, fn, tp = conf_matrix.ravel()
    
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp)

    return tp, fp, tn, fn, sens, spec, prec


def compute_micro_average(tp: List[int],
                          fp: List[int],
                          tn: List[int],
                          fn: List[int]) -> Tuple[float, float, float, float]:
    tp_mean, fp_mean, tn_mean, fn_mean = np.mean(tp), np.mean(fp), np.mean(tn), np.mean(fn)
    micro_sens = tp_mean / (tp_mean + fn_mean)
    micro_spec = tn_mean / (tn_mean + fp_mean)
    micro_prec = tp_mean / (tp_mean + fp_mean)

    micro_f1 = 2 * micro_sens * micro_prec / (micro_sens + micro_prec)

    return micro_sens, micro_spec, micro_prec, micro_f1


def compute_macro_average(sens: float,
                          spec: float,
                          prec: float) -> Tuple[float, float, float, float]:
    macro_sens = np.mean(sens)
    macro_spec = np.mean(spec)
    macro_prec = np.mean(prec)

    macro_f1 = 2 * macro_sens * macro_prec / (macro_sens + macro_prec)

    return macro_sens, macro_spec, macro_prec, macro_f1


def find_best_threshold(y_true: np.ndarray,
                        y_probs: np.ndarray,) -> np.ndarray:
    best_threshold = []

    for i in range(0, y_true.shape[1]):
        prec, sens, thresholds = precision_recall_curve(y_true[:, i], y_probs[:, i])
        f1 = 2 * sens[:-1] * prec[:-1] / (sens[:-1] + prec[:-1])

        best_idx = np.nanargmax(f1)
        best_threshold.append(thresholds[best_idx])

    return best_threshold