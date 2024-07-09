import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchaudio
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os


def load_audio(audio_path):
    y, sr = librosa.load(audio_path)
    return y, sr

def get_spectrogram(y, sr, n_fft=2048, hop_length=512):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db

def get_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def get_mfcc(y, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc

def get_chromagram(y, sr, n_fft=2048, hop_length=512):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return chroma

def get_all_features(y, sr, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=13):
    return {
        'spectrogram': get_spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length),
        'mel_spectrogram': get_mel_spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels),
        'mfcc': get_mfcc(y, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length),
        'chromagram': get_chromagram(y, sr, n_fft=n_fft, hop_length=hop_length)
    }

def get_frames(spectrogram, frame_length, hop_length):
    frames = []
    for i in range(0, spectrogram.shape[1] - frame_length + 1, hop_length):
        frame = spectrogram[:, i:i+frame_length]
        if frame.shape[1] < frame_length:
            pad_width = frame_length - frame.shape[1]
            frame = np.pad(frame, ((0, 0), (0, pad_width)), mode='constant')
        frames.append(frame)
    return np.array(frames)

def get_spectrogram_frames(y, sr, n_fft=2048, hop_length=512):
    S_db = get_spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length)
    frame_length = n_fft // 2 + 1
    return get_frames(S_db, frame_length, hop_length)

def get_mel_spectrogram_frames(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    mel_S_db = get_mel_spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    frame_length = n_mels  # mel 스펙트로그램의 경우 주파수 축이 n_mels
    return get_frames(mel_S_db, frame_length, hop_length)


def get_patch(audio_paths, patch_size=128, step=64):
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


def save_np(data, path, file_name='data.npy'):
    np.save(os.path.join(path, file_name), data)

def rename_files(path):
    print(path)
    for i, file in enumerate(os.listdir(path)):
        if file.endswith('.wav'):
            new_file_name = file[:-4]  # Remove the last 4 characters (.wav)
            os.rename(os.path.join(path, file), os.path.join(path, new_file_name))
            print(new_file_name)



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



# # 패치 생성
# patches = create_dataset([data_path])

# # 일부 패치 이미지로 저장 (예: 처음 5개)
# for i in range(min(5, len(patches))):
#     plt.figure(figsize=(3, 3))
#     plt.imshow(patches[i].squeeze(), cmap='viridis', aspect='auto')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title(f'Patch {i+1}')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f'patch_{i+1}.png'))
#     plt.close()

# # 패치 numpy 배열로 저장
# np.save(os.path.join(save_dir, 'patches.npy'), patches)

# 예제 데이터 로드 및 스펙트로그램 생성
# audio_paths = ['example1.wav', 'example2.wav']
# dataset_normal = create_dataset(audio_paths, patch_size=128, step=64)
# dataset_normal = dataset_normal.transpose(0, 3, 1, 2)  # (N, 1, H, W) 형태로 변환

