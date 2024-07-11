import torch
import torchaudio
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split



class AudioDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        self.root_dir = os.path.join(root_dir, subset)
        self.converted_dir = os.path.join(self.root_dir, 'converted')
        self.original_dir = os.path.join(self.root_dir, 'original')
        self.transform = transform
        
        self.converted_files = sorted(os.listdir(self.converted_dir))
        self.original_files = sorted(os.listdir(self.original_dir))
        
    def __len__(self):
        return len(self.converted_files) + len(self.original_files)
    
    def __getitem__(self, idx):
        if idx < len(self.converted_files):
            file_path = os.path.join(self.converted_dir, self.converted_files[idx])
            label = 1  # Converted
        else:
            idx -= len(self.converted_files)
            file_path = os.path.join(self.original_dir, self.original_files[idx])
            label = 0  # Original
        
        data = np.load(file_path)
        
        if self.transform:
            data = self.transform(data)
        
        return torch.from_numpy(data).float(), label

class AudioDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None, seed: int=42):
        if stage == 'fit' or stage is None:
            train_full = AudioDataset(self.data_dir, subset='train')
            train_size = int(0.8 * len(train_full))
            val_size = len(train_full) - train_size
            self.train_dataset, self.val_dataset = random_split(
                train_full, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(seed)  # for reproducibility
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = AudioDataset(self.data_dir, subset='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)



# class MyDataset(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# class DataModule(pl.LightningDataModule):
#     def __init__(self, train_data, val_data, test_data, batch_size=32):
#         super().__init__()
#         self.train_data = train_data
#         self.val_data = val_data
#         self.test_data = test_data
#         self.batch_size = batch_size

#     def setup(self, stage=None):
#         self.train_dataset = MyDataset(self.train_data)
#         self.val_dataset = MyDataset(self.val_data)
#         self.test_dataset = MyDataset(self.test_data)

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size)

#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size)



if __name__ == '__main__':
    # 데이터 경로 설정
    data_path = r'C:\data\Music\musdb18hq\vocals\train'
    bsro_sep_data_path = r'C:\data\Music\musdb18hq\mixture\train\BS-Roformer-1297'
    inst_sep_data_path = r'C:\data\Music\musdb18hq\mixture\train\instVoc'
    melro_sep_data_path = r'C:\data\Music\musdb18hq\mixture\train\Mel-Roformer'

    melro_covert_vocal_path = r'C:\data\Music\musdb18hq\mixture\train\Mel-Roformer\convert'
    inst_convert_vocal_path = r'C:\data\Music\musdb18hq\mixture\train\instVoc\convert'
    bsro_convert_vocal_path = r'C:\data\Music\musdb18hq\mixture\train\BS-Roformer-1297\convert'
