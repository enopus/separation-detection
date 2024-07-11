import hvplot.pandas
import pandas as pd
import numpy as np
import holoviews as hv
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os


def plot_spectrum(data, path, name, sr=44100):
    plt.figure(figsize=(10, 4))
    if 'mel' in name:
        librosa.display.specshow(data=data, sr=sr, x_axis='time', y_axis='mel')
        title = 'Mel-spectrogram'
    elif 'chromagram' in name:
        librosa.display.specshow(data=data, sr=sr, x_axis='time', y_axis='chroma')
        title = 'Chromagram'
    elif 'mfcc' in name:
        librosa.display.specshow(data=data, sr=sr, x_axis='time')
        title = 'MFCC'
    else:
        librosa.display.specshow(data=data, sr=sr, x_axis='time', y_axis='hz')
        title = 'Spectrogram'
    
    name = name.replace('.npy', '.png')
    path = os.path.join(path, 'images')
    if not os.path.exists(path):
        os.makedirs(path)
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(path, name))
    plt.close()