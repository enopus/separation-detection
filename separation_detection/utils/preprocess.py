
import os
import librosa
import numpy as np
import cv2

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

def get_frames(spectrogram, frame_length, hop_length, target_size=(256, 256)):
    frames = []
    for i in range(0, spectrogram.shape[1] - frame_length + 1, hop_length):
        frame = spectrogram[:, i:i+frame_length]
        if frame.shape[1] < frame_length:
            pad_width = frame_length - frame.shape[1]
            frame = np.pad(frame, ((0, 0), (0, pad_width)), mode='constant')
            
        # Normalize to 0-255 range for cv2
        frame_normalized = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
        
        # Resize using cv2
        frame_resized = cv2.resize(frame_normalized, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize back to 0-1 range
        frame_resized = frame_resized.astype(np.float32) / 255.0
        
        frames.append(frame_resized)
    return np.array(frames)

def get_spectrogram_frames(y, sr, n_fft=2048, hop_length=512, target_size=(256, 256)):
    S_db = get_spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length)
    frame_length = n_fft // 2 + 1
    return get_frames(S_db, frame_length, hop_length, target_size)

def get_mel_spectrogram_frames(y, sr, n_fft=2048, hop_length=512, n_mels=128, target_size=(256, 256)):
    mel_S_db = get_mel_spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    frame_length = n_mels  # mel 스펙트로그램의 경우 주파수 축이 n_mels
    return get_frames(mel_S_db, frame_length, hop_length, target_size)


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

