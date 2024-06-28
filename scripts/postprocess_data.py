import os
import shutil
import soundfile as sf
import librosa
import numpy as np


def apply_zero_padding(audio_data, max_length):
    return np.pad(audio_data, (0, max_length - len(audio_data)), 'constant')

def preprocess_audio_files(file_path):
    max_length = 0
    
    for file in os.listdir(file_path):
        audio_data, _ = sf.read(os.path.join(file_path, file))
        max_length = max(max_length, len(audio_data))
    
    for file in os.listdir(file_path):    
        audio_data, _ = sf.read(os.path.join(file_path, file))
        padded_audio = apply_zero_padding(audio_data, max_length)
        processed_file_name = file.replace('.wav', '_processed.wav')
        sf.write(os.path.join(file_path, processed_file_name), padded_audio, _)

path_train = r'C:\data\Music\musdb18hq\mixture\train'
path_test = r'C:\data\Music\musdb18hq\mixture\test'

preprocess_audio_files(path_train)
preprocess_audio_files(path_test)

