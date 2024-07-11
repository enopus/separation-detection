import torch
import torchaudio
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import cv2


import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size=32):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = MyDataset(self.train_data)
        self.val_dataset = MyDataset(self.val_data)
        self.test_dataset = MyDataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


# 데이터 로딩 및 전처리
class VocalDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, labels):
        self.file_list = file_list
        self.labels = labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.file_list[idx])
        spectrogram = torchaudio.transforms.Spectrogram()(waveform)
        return spectrogram, self.labels[idx]




# 데이터 경로 설정
data_path = r'C:\data\Music\musdb18hq\vocals\train'
bsro_sep_data_path = r'C:\data\Music\musdb18hq\mixture\train\BS-Roformer-1297'
inst_sep_data_path = r'C:\data\Music\musdb18hq\mixture\train\instVoc'
melro_sep_data_path = r'C:\data\Music\musdb18hq\mixture\train\Mel-Roformer'

melro_covert_vocal_path = r'C:\data\Music\musdb18hq\mixture\train\Mel-Roformer\convert'
inst_convert_vocal_path = r'C:\data\Music\musdb18hq\mixture\train\instVoc\convert'
bsro_convert_vocal_path = r'C:\data\Music\musdb18hq\mixture\train\BS-Roformer-1297\convert'
