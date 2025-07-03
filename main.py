import pandas as pd
import torch
import torch.nn as nn

from AnalysisData import ECGDataset, get_mean_std, Compose, ToTensor, Normalize
from Model import ResNet, Bottleneck, train, test, device

def process_metadata(dataset):
    metadata = dataset[['age', 'sex', 'height', 'weight']].copy()

    with pd.option_context("future.no_silent_downcasting", True):
        metadata['age']    = metadata['age'].fillna(metadata['age'].median())
        metadata['height'] = metadata['height'].fillna(metadata['height'].median())
        metadata['weight'] = metadata['weight'].fillna(metadata['weight'].median())

    return metadata.values


path = "data/physionet.org/files/ptb-xl/1.0.1/"
sampling_rate = 100

target_labels = {
    'sinus': ['NORM', 'SR'],
    'arr': ['SARRH', 'SVARR'],
    'tach': ['STACH', 'SVATC', 'PSVT'],
    'brad': ['SBRAD'],
    'afib': ['AFIB', 'AFLT']
}

valid_fold = 9
test_fold = 10
use_metadata = False
use_pqrst = False

ecg_dataset = ECGDataset(path, 
                         target_labels, 
                         sampling_rate,  
                         use_pqrst=use_pqrst, 
                         process_metadata=None,
                         transform=None)

mean_ecg, std_ecg = get_mean_std(ecg_dataset.get_ecg_signals(), axis=(0, 1))
mean_meta, std_meta, mean_pqrst, std_pqrst = None, None, None, None

if use_metadata:
    mean_meta, std_meta = get_mean_std(ecg_dataset.get_metadata())
if use_pqrst:
    mean_pqrst, std_pqrst = get_mean_std(ecg_dataset.get_pqrst_features())

ecg_dataset.transform = Compose([ToTensor(), Normalize(mean_ecg, std_ecg, mean_meta, std_meta, mean_pqrst, std_pqrst)])

train_dataset, val_dataset, test_dataset = ecg_dataset.get_dataset(valid_fold=valid_fold, test_fold=test_fold)
ecg_dataset.close_dataset()

batch_size    = 32
learning_rate = 0.001
n_epoch       = 15
num_classes   = len(target_labels)

treshold_preds = 0.5

criterion = nn.BCEWithLogitsLoss()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size)

net = ResNet(Bottleneck, [2, 2, 2, 2], num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

net, train_history, val_history = train(net, train_loader, val_loader, 
                                        n_epoch, optimizer, criterion, treshold_preds)
test_metrics = test(net, test_loader, criterion, treshold_preds)

print(f"Test metrics: {test_metrics}")