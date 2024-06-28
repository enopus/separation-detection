import hvplot.pandas
import pandas as pd
import numpy as np
import holoviews as hv
import librosa

hv.extension('bokeh')

def plot_spectrogram(y, sr, title):
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    df = pd.DataFrame(S_db, columns=np.arange(S_db.shape[1]))
    df['freq'] = librosa.fft_frequencies(sr=sr)
    df = df.melt(id_vars=['freq'], var_name='time', value_name='db')
    plot = df.hvplot.heatmap(
        x='time', y='freq', C='db', colormap='viridis',
        title=title, width=800, height=400,
        xlabel='Time', ylabel='Frequency (Hz)',
        clabel='Amplitude (dB)'
    )
    return plot

def plot_melspectrogram(y, sr, title):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    df = pd.DataFrame(S_db, columns=np.arange(S_db.shape[1]))
    df['mel'] = librosa.mel_frequencies(n_mels=S_db.shape[0], sr=sr)
    df = df.melt(id_vars=['mel'], var_name='time', value_name='db')
    plot = df.hvplot.heatmap(
        x='time', y='mel', C='db', colormap='viridis',
        title=title, width=800, height=400,
        xlabel='Time', ylabel='Mel Frequency',
        clabel='Power (dB)'
    )
    return plot

def plot_waveform(y, sr, title):
    times = np.linspace(0, len(y)/sr, num=len(y))
    df = pd.DataFrame({'time': times, 'amplitude': y})
    plot = df.hvplot.line(
        x='time', y='amplitude',
        title=title, width=800, height=200,
        xlabel='Time', ylabel='Amplitude'
    )
    return plot

def plot_chromagram(y, sr, title):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    df = pd.DataFrame(chroma, columns=np.arange(chroma.shape[1]))
    df['pitch_class'] = np.arange(12)
    df = df.melt(id_vars=['pitch_class'], var_name='time', value_name='energy')
    plot = df.hvplot.heatmap(
        x='time', y='pitch_class', C='energy', colormap='viridis',
        title=title, width=800, height=400,
        xlabel='Time', ylabel='Pitch Class',
        clabel='Energy'
    )
    return plot

def plot_mfcc(y, sr, title):
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    df = pd.DataFrame(mfcc, columns=np.arange(mfcc.shape[1]))
    df['mfcc'] = np.arange(mfcc.shape[0])
    df = df.melt(id_vars=['mfcc'], var_name='time', value_name='coefficient')
    plot = df.hvplot.heatmap(
        x='time', y='mfcc', C='coefficient', colormap='viridis',
        title=title, width=800, height=400,
        xlabel='Time', ylabel='MFCC',
        clabel='Coefficient'
    )
    return plot


def compare_audio_files(*args, **kwargs):
    plot_types = kwargs.get('plot_types', ['spectrogram', 'melspectrogram', 'waveform', 'mfcc', 'chromagram'])
    
    plots = {}
    for plot_type in plot_types:
        plots[plot_type] = []

    for file, title in args:
        y, sr = librosa.load(file)
        
        for plot_type in plot_types:
            if plot_type == 'spectrogram':
                plots[plot_type].append(plot_spectrogram(y, sr, f'Spectrogram - {title}'))
            elif plot_type == 'melspectrogram':
                plots[plot_type].append(plot_melspectrogram(y, sr, f'Mel Spectrogram - {title}'))
            elif plot_type == 'waveform':
                plots[plot_type].append(plot_waveform(y, sr, f'Waveform - {title}'))
            elif plot_type == 'mfcc':
                plots[plot_type].append(plot_mfcc(y, sr, f'MFCC - {title}'))
            elif plot_type == 'chromagram':
                plots[plot_type].append(plot_chromagram(y, sr, f'Chromagram - {title}'))

    layout = hv.Layout([hv.Column(plot_list) for plot_list in plots.values()]).cols(1)
    return layout

# 사용 예시
comparison = compare_audio_files(
    ('original_mixture.wav', 'Original Mixture'),
    ('separated_vocal.wav', 'Separated Vocal'),
    ('converted_vocal.wav', 'Converted Vocal'),
    plot_types=['spectrogram', 'melspectrogram', 'waveform']
)
comparison

