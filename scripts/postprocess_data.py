import os
import shutil
import soundfile as sf
import librosa
import numpy as np
import separation_detection.utils.visualization as visualization


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

def spectrum_to_image(file_path):
    for file in os.listdir(file_path):
        data = np.load(os.path.join(file_path, file))
        visualization.plot_spectrum(data=data, path=file_path, name=file)


original_train_spectrogram_path = r'C:\data\Music\musdb18hq\result\train\original\spectrogram'
original_train_mel_spectrogram_path = r'C:\data\Music\musdb18hq\result\train\original\mel_spectrogram'
original_train_mfcc_path = r'C:\data\Music\musdb18hq\result\train\original\mfcc'
original_train_chromagram_path = r'C:\data\Music\musdb18hq\result\train\original\chromagram'



original_test_spectrogram_path = r'C:\data\Music\musdb18hq\result\test\original\spectrogram'
original_test_mel_spectrogram_path = r'C:\data\Music\musdb18hq\result\test\original\mel_spectrogram'
original_test_mfcc_path = r'C:\data\Music\musdb18hq\result\train\original\mfcc'
original_test_chromagram_path = r'C:\data\Music\musdb18hq\result\train\original\chromagram'


sep_train_spectrogram_path = r'C:\data\Music\musdb18hq\result\train\sep\spectrogram'
sep_train_mel_spectrogram_path = r'C:\data\Music\musdb18hq\result\train\sep\mel_spectrogram'
sep_train_mfcc_path = r'C:\data\Music\musdb18hq\result\train\sep\mfcc'
sep_train_chromagram_path = r'C:\data\Music\musdb18hq\result\train\sep\chromagram'


convert_train_spectrogram_path = r'C:\data\Music\musdb18hq\result\train\convert\spectrogram'
convert_train_mel_spectrogram_path = r'C:\data\Music\musdb18hq\result\train\convert\mel_spectrogram'
convert_train_mfcc_path = r'C:\data\Music\musdb18hq\result\train\convert\mfcc'
convert_train_chromagram_path = r'C:\data\Music\musdb18hq\result\train\convert\chromagram'



spectrum_to_image(original_train_spectrogram_path)
spectrum_to_image(original_train_mel_spectrogram_path)
print('Converting original_train End')

spectrum_to_image(original_test_spectrogram_path)
spectrum_to_image(original_test_mel_spectrogram_path)
print('Converting original_test End')

spectrum_to_image(sep_train_spectrogram_path)
spectrum_to_image(sep_train_mel_spectrogram_path)

print('Converting sep_train End')


spectrum_to_image(convert_train_spectrogram_path)
spectrum_to_image(convert_train_mel_spectrogram_path)

print('Converting convert_train End')