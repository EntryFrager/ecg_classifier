import numpy as np
import pandas as pd
import ast
import wfdb
import torch
import torchvision
from torch.utils.data import Dataset


class ECGDataset:
    def __init__(self, path = 'data/physionet.org/files/ptb-xl/1.0.1/', sampling_rate = 100):
        self.path = path
        self.sampling_rate = sampling_rate

        self.scp_statements = pd.read_csv(path + 'scp_statements.csv', index_col = 0).reset_index().rename(columns={'index': 'scp_code'})
        self.ptbxl_dataset = pd.read_csv(path + 'ptbxl_database.csv', index_col = 'ecg_id')

        self.valid_fold = 9
        self.test_fold = 10
 
        self.target_codes   = None
        self.metadata_features = ['age', 'sex', 'height', 'weight', 'heart_axis', 'infarction_stadium1', 'infarction_stadium2', 'strat_fold']

        self.ecg_signals    = None
        self.pqrst_features = None
        self.metadata       = None
        self.labels         = None


    def proccess_data(self, target_codes, check_pqrst=False):
        self._set_target_codes(target_codes)
        self._load_data()
        self.metadata = self._proccess_metadata()
        
        if (check_pqrst):
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

        unique_value_infrstd = {np.nan: 0,
                                'unknown': -1,
                                'Stadium I': 1,
                                'Stadium I-II': 1.5,
                                'Stadium II': 2,
                                'Stadium II-III': 2.5,
                                'Stadium III': 3,}
        fields_name = ['infarction_stadium1', 'infarction_stadium2']

        for field_name in fields_name:
            metadata[field_name] = metadata[field_name].map(unique_value_infrstd)

        with pd.option_context("future.no_silent_downcasting", True):
            metadata['age'] = metadata['age'].fillna(metadata['age'].mean())
            metadata['height'] = metadata['height'].fillna(metadata['height'].median())
            metadata['weight'] = metadata['weight'].fillna(metadata['weight'].median())

        unique_value_hrtax = {np.nan: -10000,
                              'AXL': -3,
                              'ALAD': -2,
                              'LAD': -1,
                              'MID': 0,
                              'RAD': 1,
                              'ARAD': 2,
                              'AXR': 3,
                              'SAG': 10000,}
        metadata['heart_axis'] = metadata['heart_axis'].map(unique_value_hrtax)

        return metadata
    
    
    def _proccess_pqrst_features(self):

        return
    

    def get_dataset(self):
        train_mask = (self.ptbxl_dataset.strat_fold != self.valid_fold) & (self.ptbxl_dataset.strat_fold != self.test_fold)
        val_mask   = (self.ptbxl_dataset.strat_fold == self.valid_fold)
        test_mask  = (self.ptbxl_dataset.strat_fold == self.test_fold)

        train = {'ecg_signals': self.ecg_signals[train_mask],
                 'metadata': self.metadata.values[train_mask],
                 'labels': self.labels.values[train_mask]}
        
        val = {'ecg_signals': self.ecg_signals[val_mask],
               'metadata': self.metadata.values[val_mask],
               'labels': self.labels.values[val_mask]}
        
        test = {'ecg_signals': self.ecg_signals[test_mask],
               'metadata': self.metadata.values[test_mask],
               'labels': self.labels.values[test_mask]}
        
        if self.pqrst_features is not None:
            train['pqrst_features'] = self.pqrst_features[train_mask]
            val['pqrst_features']   = self.pqrst_features[val_mask]
            test['pqrst_features']  = self.pqrst_features[test_mask]

        
        return train, val, test
    

class ECGTorchDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.ecg_signals     = torch.FloatTensor(dataset['ecg_signals'])
        self.metadata       = torch.FloatTensor(dataset['metadata'])
        self.pqrst_features = torch.FloatTensor(dataset['pqrst_features']) if 'pqrst_features' in dataset else None
        self.labels         = torch.FloatTensor(dataset['labels'])
        self.transform = transform

    
    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        dataset =  {'ecg_signals': self.ecg_signals[index],
                    'metadata': self.metadata[index],
                    'labels': self.labels[index]}
        
        if self.pqrst_features is not None:
            dataset['pqrst_features'] = self.pqrst_features[index]

        if self.transform is not None:
            dataset = self.transform(dataset)

        return dataset
    

class Normalize:
    def __init__(self):
        self.ecg_signals_stats     = None
        self.metadata_stats       = None
        self.pqrst_features_stats = None


    def get_stats(self, dataset):
        self.ecg_signals_stats = {'mean': np.mean(dataset['ecg_signals']),
                                 'std': np.std(dataset['ecg_signals'])}
        
        self.metadata_stats = {'mean': np.mean(dataset['metadata']),
                               'std': np.std(dataset['metadata'])}
        
        if 'pqrst_features' in dataset:
            self.pqrst_features_stats = {'mean': np.mean(dataset['pqrst_features']),
                                         'std': np.std(dataset['pqrst_features'])}
            

    def norm(self, dataset):
        dataset['ecg_signals'] = (dataset['ecg_singals'] - self.ecg_signals_stats['mean']) / self.ecg_signals_stats['std']
        
        dataset['metadata'] = (dataset['metadata'] - self.metadata_stats['mean']) / self.metadata_stats['std']
        
        if self.pqrst_features_stats is not None:
            dataset['pqrst_features'] = (dataset['pqrst_features'] - self.pqrst_features_stats['mean']) / self.pqrst_features_stats['std']

        return dataset


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

transform = Normalize()
transform.get_stats(train)

train = ECGTorchDataset(train, transform=transform)
val = ECGTorchDataset(val)
test = ECGTorchDataset(test)

train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val, batch_size=32, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)

