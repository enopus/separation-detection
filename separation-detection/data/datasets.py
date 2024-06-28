import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchaudio
import numpy as np
import librosa

def get_spectrogram(audio_path, n_fft=2048, hop_length=512):
    y, sr = librosa.load(audio_path)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db, sr

def create_dataset(audio_paths, patch_size=128, step=64):
    patches = []
    for audio_path in audio_paths:
        spectrogram, sr = get_spectrogram(audio_path)
        for i in range(0, spectrogram.shape[0] - patch_size + 1, step):
            for j in range(0, spectrogram.shape[1] - patch_size + 1, step):
                patch = spectrogram[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
    patches = np.array(patches)
    patches = patches[..., np.newaxis]  # 채널 추가
    return patches

# 예제 데이터 로드 및 스펙트로그램 생성
audio_paths = ['example1.wav', 'example2.wav']
dataset_normal = create_dataset(audio_paths, patch_size=128, step=64)
dataset_normal = dataset_normal.transpose(0, 3, 1, 2)  # (N, 1, H, W) 형태로 변환

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