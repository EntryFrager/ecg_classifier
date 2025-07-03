import os
import numpy as np
import pandas as pd
import wfdb
import ast
import torch
from torch.utils.data import Dataset, Subset


class ECGDataset(Dataset):
    def __init__(self, 
                 path='data/physionet.org/files/ptb-xl/1.0.1/',  
                 target_labels=None, 
                 sampling_rate=100,
                 use_pqrst=False,
                 process_metadata=None,
                 transform=None):
        self.path = path
        self.sampling_rate = sampling_rate

        self.scp_statements = pd.read_csv(path + 'scp_statements.csv', index_col = 0).reset_index().rename(columns={'index': 'scp_code'})
        self.ptbxl_dataset  = pd.read_csv(path + 'ptbxl_database.csv', index_col = 'ecg_id')
 
        self.target_labels = target_labels

        self.labels         = self._set_target_labels()
        self.ecg_signals    = self._process_ecg_signals()
        self.pqrst_features = self._process_pqrst_features() if use_pqrst else None
        self.metadata       = process_metadata(self.ptbxl_dataset) if process_metadata is not None else None
        self.transform      = transform

    
    def get_dataset(self, valid_fold=9, test_fold=10):
        train_idx = np.where((self.ptbxl_dataset.strat_fold != valid_fold) & (self.ptbxl_dataset.strat_fold != test_fold))[0]
        val_idx   = np.where((self.ptbxl_dataset.strat_fold == valid_fold))[0]
        test_idx  = np.where((self.ptbxl_dataset.strat_fold == test_fold))[0]

        train = Subset(self, train_idx)
        val   = Subset(self, val_idx)
        test  = Subset(self, test_idx)
        
        return train, val, test


    def _set_target_labels(self):
        self.ptbxl_dataset['scp_codes'] = self.ptbxl_dataset['scp_codes'].apply(lambda x: ast.literal_eval(x))

        for label, target_labels in self.target_labels.items():
            target_values = np.zeros(len(self.ptbxl_dataset), dtype=int)
            for target_label in target_labels:
                target_values += self.ptbxl_dataset['scp_codes'].apply(lambda x: 1 if (target_label in x and (x[target_label] >= 50 or x[target_label] == 0)) else 0).to_numpy(dtype=int)
            self.ptbxl_dataset[label] = np.where(target_values >= 1, 1, 0)
        
        return self.ptbxl_dataset[self.target_labels.keys()].values


    def _process_ecg_signals(self):
        if self.sampling_rate == 100:
            files = [filename for filename in self.ptbxl_dataset['filename_lr']]
        elif self.sampling_rate == 500:
            files = [filename for filename in self.ptbxl_dataset['filename_hr']]
        
        return np.array([wfdb.rdsamp(self.path + file)[0] for file in files])

        
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
    

    def get_ptbxl_dataset(self):
        return self.ptbxl_dataset
    
    
    def get_scp_statements(self):
        return self.scp_statements
    

    def get_label(self):
        return self.labels
    

    def get_ecg_signals(self):
        return self.ecg_signals
    

    def get_pqrst_features(self):
        return self.pqrst_features
    

    def get_metadata(self):
        return self.metadata


    def close_dataset(self):
        del self.ptbxl_dataset
        del self.scp_statements

    
    def save_csv(self):
        os.makedirs('./ParsData/', exist_ok=True)

        if self.ecg_signals is not None:
            self.ecg_signals.to_csv('./ParsData/ecg_signals.csv')
        
        if self.metadata is not None:
            self.metadata.to_csv('./ParsData/metadata.csv')
        
        if self.pqrst_features is not None:
            self.pqrst_features.to_csv('./ParsData/pqrst_features.csv')


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
    def __init__(self, mean_ecg, std_ecg, mean_meta, std_meta, mean_pqrst, std_pqrst):
        super().__init__()
        self.mean_ecg = mean_ecg
        self.std_ecg  = std_ecg

        self.mean_meta = mean_meta
        self.std_meta  = std_meta

        self.mean_pqrst = mean_pqrst
        self.std_pqrst  = std_pqrst


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


