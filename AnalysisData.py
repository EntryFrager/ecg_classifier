import numpy as np
import pandas as pd
import ast
import wfdb
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class ECGDataset:
    def __init__(self, path = 'data/physionet.org/files/ptb-xl/1.0.1/', sampling_rate = 100):
        self.path = path
        self.sampling_rate = sampling_rate

        self.scp_statements = pd.read_csv(path + 'scp_statements.csv', index_col = 0).reset_index().rename(columns={'index': 'scp_code'})
        self.ptbxl_dataset = pd.read_csv(path + 'ptbxl_database.csv', index_col = 'ecg_id')

        self.valid_fold = 9
        self.test_fold = 10
 
        self.target_codes   = None
        self.metadata_features = ['age', 'sex', 'height', 'weight', 'heart_axis', 'infarction_stadium1', 'infarction_stadium2']

        self.ecg_signals    = None
        self.pqrst_features = None
        self.metadata       = None
        self.labels         = None


    def proccess_data(self, target_codes, use_metadata=False, use_pqrst=False):
        self._set_target_codes(target_codes)
        self._load_data()

        if use_metadata:
            self.metadata = self._proccess_metadata()
        
        if use_pqrst:
            self.pqrst_features = self._proccess_pqrst_features()


    def _set_target_codes(self, target_codes):
        self.target_codes = target_codes

        self.ptbxl_dataset['scp_codes'] = self.ptbxl_dataset['scp_codes'].apply(lambda x: ast.literal_eval(x))

        for label, codes in self.target_codes.items():
            self.ptbxl_dataset[label] = self.ptbxl_dataset['scp_codes'].apply(lambda x: 1 if any(i in x for i in codes) else 0)

        sum_targets = self.ptbxl_dataset[list(self.target_codes.keys())].sum(axis = 1)
        self.ptbxl_dataset = self.ptbxl_dataset.drop(self.ptbxl_dataset[sum_targets == 0].index)

        self.labels = self.ptbxl_dataset[list(self.target_codes.keys())]


    def _load_data(self):
        if self.sampling_rate == 100:
            files = [filename for filename in self.ptbxl_dataset['filename_lr']]
        elif self.sampling_rate == 500:
            files = [filename for filename in self.ptbxl_dataset['filename_hr']]
        
        self.ecg_signals = [wfdb.rdsamp(self.path + file)[0] for file in files]
        self.ecg_signals = np.array(self.ecg_signals)


    def _proccess_metadata(self):
        metadata = self.ptbxl_dataset[self.metadata_features].copy()

        metadata = pd.get_dummies(metadata, columns=['infarction_stadium1', 'infarction_stadium2', 'heart_axis'])

        with pd.option_context("future.no_silent_downcasting", True):
            metadata['age'] = metadata['age'].fillna(metadata['age'].median())
            metadata['height'] = metadata['height'].fillna(metadata['height'].median())
            metadata['weight'] = metadata['weight'].fillna(metadata['weight'].median())

        return metadata
    
    
    def _proccess_pqrst_features(self):

        return
    

    def get_dataset(self):
        train_mask = (self.ptbxl_dataset.strat_fold != self.valid_fold) & (self.ptbxl_dataset.strat_fold != self.test_fold)
        val_mask   = (self.ptbxl_dataset.strat_fold == self.valid_fold)
        test_mask  = (self.ptbxl_dataset.strat_fold == self.test_fold)

        train = {'ecg_signals': self.ecg_signals[train_mask],
                 'labels': self.labels.values[train_mask]}
        
        val = {'ecg_signals': self.ecg_signals[val_mask],
               'labels': self.labels.values[val_mask]}
        
        test = {'ecg_signals': self.ecg_signals[test_mask],
               'labels': self.labels.values[test_mask]}
        
        if self.metadata is not None:
            train['metadata'] = self.metadata.values[train_mask]
            val['metadata']   = self.metadata.values[val_mask]
            test['metadata']  = self.metadata.values[test_mask]
        
        if self.pqrst_features is not None:
            train['pqrst_features'] = self.pqrst_features[train_mask]
            val['pqrst_features']   = self.pqrst_features[val_mask]
            test['pqrst_features']  = self.pqrst_features[test_mask]

        
        return train, val, test


class ECGTorchDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.ecg_signals    = dataset['ecg_signals']
        self.metadata       = dataset['metadata'] if 'metadata' in dataset else None
        self.pqrst_features = dataset['pqrst_features'] if 'pqrst_features' in dataset else None
        self.labels         = dataset['labels']
        self.transform = transform


    def __getitem__(self, index):
        sample = {'ecg_signals': self.ecg_signals[index],
                  'labels': self.labels[index]}
        
        if self.metadata is not None:
            sample['metadata'] = self.metadata[index]

        if self.pqrst_features is not None:
            sample['pqrst_features'] = self.pqrst_features[index]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
    
    
    def __len__(self):
        return len(self.labels)
    

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms


    def __call__(self, obj):
        for t in self.transforms:
            obj = t(obj)

        return obj


class ToTensor:
    def __init__(self):
        pass
    
    
    def __call__(self, sample):
        for key in sample:
            sample[key] = torch.FloatTensor(sample[key])
        
        return sample


class Normalize(torch.nn.Module):
    def __init__(self, mean_ecg, std_ecg, mean_meta=None, std_meta=None, mean_pqrst=None, std_pqrst=None):
        self.mean_ecg = mean_ecg
        self.std_ecg  = std_ecg

        self.mean_meta = mean_meta
        self.std_meta  = std_meta

        self.mean_pqrst = mean_pqrst
        self.std_pqrst  = std_pqrst


    def forward(self, sample):
        sample['ecg_signals'] = (sample['ecg_signals'] - mean_ecg) / std_ecg

        if mean_meta is not None:
            sample['meta'] = (sample['meta'] - mean_meta) / std_meta

        if mean_pqrst is not None:
            sample['pqrst_features'] = (sample['pqrst_features'] - mean_pqrst) / std_pqrst

        return sample
    

def get_mean_std(value, axis=None):
    return value.mean(axis=axis), value.std(axis=axis)


sampling_rate = 100
path = "data/physionet.org/files/ptb-xl/1.0.1/"

target_codes = {'NORM': ['SR'],
                'ARR': ['AFLT', 'SVARR', 'SARRH'],
                'BRAD': ['SBRAD'],
                'TACH': ['STACH', 'VTACH', 'SVTAC', 'PSVT'],
                'AFIB': ['AFIB']}

ecg_dataset = ECGDataset(path, sampling_rate)
ecg_dataset.proccess_data(target_codes)
train, val, test = ecg_dataset.get_dataset()

mean_ecg, std_ecg = get_mean_std(train['ecg_signals'], axis=(0, 1))
mean_meta, std_meta, mean_pqrst, std_pqrst = None, None, None, None

if 'metadata' in train:
    mean_meta, std_meta = get_mean_std(train['metadata'])

if 'pqrst_features' in train:
    mean_pqrst, std_pqrst = get_mean_std(train['pqrst_features'])

transform = Compose([ToTensor(), Normalize(mean_ecg, std_ecg, mean_meta, std_meta, mean_pqrst, std_pqrst)])

train = ECGTorchDataset(train, transform=transform)
val   = ECGTorchDataset(val)
test  = ECGTorchDataset(test)

train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val,   batch_size=32, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test,  batch_size=32, shuffle=True)