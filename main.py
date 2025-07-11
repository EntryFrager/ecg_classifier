import torch
import torch.nn as nn

from ecg_classifier import ECGDataset, ResNet, BasicBlock, train, test, device

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

train_dataset, val_dataset, test_dataset = ecg_dataset.get_dataset()
pos_weight = ecg_dataset.get_pos_weight().to(device)
ecg_dataset.close_dataset()

batch_size    = 32
learning_rate = 0.000005
n_epoch       = 30
num_classes   = len(target_labels)

patience   = 5
loss_delta = 0.17
acc_delta  = 0.15

threshold_preds = [0.5] * len(target_labels)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size)

net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

net, threshold_preds, train_history, val_history = train(net, train_loader, val_loader, 
                                                         n_epoch, optimizer, criterion, 
                                                         threshold_preds, patience, loss_delta, acc_delta)
test_loss = test(net, test_loader, criterion, threshold_preds)
