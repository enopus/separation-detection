import hvplot.pandas
import pandas as pd
import numpy as np
import holoviews as hv
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os



def plot_spectrum(data, sr, path, name='spectrum.png', title='Spectrogram'):
    plt.figure(figsize=(10, 4))
    if name in 'mel':
        librosa.display.specshow(data=data, sr=sr, x_axis='time', y_axis='mel')
    else:
        librosa.display.specshow(data=data, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(path, name))
