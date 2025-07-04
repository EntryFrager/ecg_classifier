import os
import numpy as np
import pandas as pd
import wfdb
import ast
import torch
from torch.utils.data import Dataset, Subset


class ECGDataset(Dataset):
    ecg_stat = {
        'mean': torch.FloatTensor([-0.0018, -0.0013,  0.0005,  0.0016, -0.0011, -0.0004,  0.0002, -0.0009, -0.0015, -0.0017, -0.0008, -0.0021]),
        'std': torch.FloatTensor([[0.1640, 0.1647, 0.1713, 0.1403, 0.1461, 0.1466, 0.2337, 0.3377, 0.3336, 0.3058, 0.2731, 0.2755]])
    }
    
    metadata_stat = {
        'mean': torch.tensor(74.2453, dtype=torch.float32),
        'std': torch.tensor(60.3269, dtype=torch.float32)
    }
    
    pqrst_stat = {
        'mean': 0,
        'std': 0
    }

    valid_fold = 9
    test_fold  = 10

    def __init__(self, 
                 path='data/physionet.org/files/ptb-xl/1.0.1/',  
                 target_labels=None, 
                 sampling_rate=100,
                 use_pqrst=False,
                 use_metadata=None):
        self.path = path
        self.sampling_rate = sampling_rate

        self.scp_statements = pd.read_csv(path + 'scp_statements.csv', index_col = 0).reset_index().rename(columns={'index': 'scp_code'})
        self.ptbxl_dataset  = pd.read_csv(path + 'ptbxl_database.csv', index_col = 'ecg_id')
 
        self.target_labels = target_labels

        self.labels         = self._set_target_labels()
        self.ecg_signals    = self._process_ecg_signals()
        self.pqrst_features = self._process_pqrst_features() if use_pqrst else None
        self.metadata       = self._process_metadata() if use_metadata else None
        self.transform      = Compose([ToTensor(), Normalize(self.ecg_stat, self.metadata_stat, self.pqrst_stat)])


    def get_dataset(self):
        train_idx = np.where((self.ptbxl_dataset.strat_fold != self.valid_fold) & (self.ptbxl_dataset['strat_fold'] != self.test_fold))[0]
        val_idx   = np.where((self.ptbxl_dataset.strat_fold == self.valid_fold))[0]
        test_idx  = np.where((self.ptbxl_dataset.strat_fold == self.test_fold))[0]

        train = Subset(self, train_idx)
        val   = Subset(self, val_idx)
        test  = Subset(self, test_idx)
        
        return train, val, test


    def _set_target_labels(self):
        self.ptbxl_dataset['scp_codes'] = self.ptbxl_dataset['scp_codes'].apply(lambda x: ast.literal_eval(x))

        for label, target_labels in self.target_labels.items():
            target_values = np.zeros(len(self.ptbxl_dataset), dtype=int)
            for target_label in target_labels:
                target_values += self.ptbxl_dataset['scp_codes'].apply(lambda x: 1 if (target_label in x and (x[target_label] >= 50 or x[target_label] == 0)) else 0).to_numpy(dtype=int) #
            self.ptbxl_dataset[label] = np.where(target_values >= 1, 1, 0)
        
        return self.ptbxl_dataset[self.target_labels.keys()].values


    def _process_ecg_signals(self):
        if self.sampling_rate == 100:
            files = [filename for filename in self.ptbxl_dataset['filename_lr']]
        elif self.sampling_rate == 500:
            files = [filename for filename in self.ptbxl_dataset['filename_hr']]
        
        return np.array([wfdb.rdsamp(self.path + file)[0] for file in files])
    

    def _process_metadata(self):
        metadata = self.ptbxl_dataset[['age', 'sex', 'height', 'weight']].copy()

        with pd.option_context("future.no_silent_downcasting", True):
            metadata['age']    = metadata['age'].fillna(metadata['age'].median())
            metadata['height'] = metadata['height'].fillna(metadata['height'].median())
            metadata['weight'] = metadata['weight'].fillna(metadata['weight'].median())

        return metadata.values


        
    def _process_pqrst_features(self):
        # in develop
        return
    
    
    def __getitem__(self, index):
        sample = {'ecg_signals': self.ecg_signals[index],
                  'labels': self.labels[index]}

        if self.metadata is not None:
            sample['metadata'] = self.metadata[index]

        if self.pqrst_features is not None:
            sample['pqrst_features'] = self.pqrst_features[index]

        if self.transform is not None:
            sample = self.transform(sample)

        sample['ecg_signals'] = sample['ecg_signals'].transpose(0, 1)

        return sample
    
    
    def __len__(self):
        return len(self.labels)
    

    def get_pos_weight(self):
        pos_weight = []

        train_mask = (self.ptbxl_dataset.strat_fold != self.valid_fold) & (self.ptbxl_dataset['strat_fold'] != self.test_fold)
        train_label = self.labels[train_mask]

        for key, _ in self.target_labels.items():
            unique_value = train_label[key].value_counts()
            pos_weight.append(unique_value['0'] / unique_value['1'])

        return pos_weight


    def close_dataset(self):
        del self.ptbxl_dataset
        del self.scp_statements


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
        for key, _ in sample.items():
            sample[key] = torch.tensor(sample[key], dtype=torch.float32)
        
        return sample


class Normalize(torch.nn.Module):
    def __init__(self, ecg_stat, metadata_stat, pqrst_stat):
        super().__init__()

        self.mean_ecg = ecg_stat['mean']
        self.std_ecg  = ecg_stat['std']

        self.mean_meta = metadata_stat['mean']
        self.std_meta  = metadata_stat['std']

        self.mean_pqrst = pqrst_stat['mean']
        self.std_pqrst  = pqrst_stat['std']


    def forward(self, sample):
        sample['ecg_signals'] = (sample['ecg_signals'] - self.mean_ecg) / self.std_ecg

        if self.mean_meta is not None:
            sample['metadata'] = (sample['metadata'] - self.mean_meta) / self.std_meta

        if self.mean_pqrst is not None:
            sample['pqrst_features'] = (sample['pqrst_features'] - self.mean_pqrst) / self.std_pqrst

        return sample


def get_mean_std(value, axis=None):
    return torch.tensor(value.mean(axis=axis).astype(np.float32), dtype=torch.float32), \
           torch.tensor(value.std(axis=axis).astype(np.float32), dtype=torch.float32)


