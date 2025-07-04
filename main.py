import torch
import torch.nn as nn

from AnalysisData import ECGDataset
from Model import ResNet, Bottleneck, train, test, device


def get_stat(dataset, target_labels):
    for key, _ in target_labels.items():
        label = dataset[key].value_counts()
        print(f'{key} unique labels: {label}')


path = "data/physionet.org/files/ptb-xl/1.0.1/"
sampling_rate = 100

target_labels = {
    'sinus': ['NORM', 'SR'],
    'arrit': ['SARRH', 'SVARR'],
    'tach': ['STACH', 'SVATC', 'PSVT'],
    'brad': ['SBRAD'],
    'afib': ['AFIB', 'AFLT']
}

use_metadata = False
use_pqrst = False

ecg_dataset = ECGDataset(path, 
                         target_labels, 
                         sampling_rate, 
                         use_pqrst=use_pqrst, 
                         use_metadata=use_metadata)

ptbxl_dataset = ecg_dataset.ptbxl_dataset
get_stat(ptbxl_dataset, target_labels)

train_dataset, val_dataset, test_dataset = ecg_dataset.get_dataset()
pos_weight = ecg_dataset.get_pos_weight()
ecg_dataset.close_dataset()

batch_size    = 32
learning_rate = 0.0001
n_epoch       = 10
num_classes   = len(target_labels)

treshold_preds = 0.5

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size)

net = ResNet(Bottleneck, [2, 2, 2, 2], num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

net, loss_train_history, loss_val_history = train(net, train_loader, val_loader, 
                                                  n_epoch, optimizer, criterion, treshold_preds)
test_metrics = test(net, test_loader, criterion, treshold_preds)

print(f"Test metrics: {test_metrics}")