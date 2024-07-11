import os
import shutil
import soundfile as sf
import librosa
import numpy as np
import cv2
import separation_detection
import separation_detection.data.datasets as datasets

def rename_file(file_path):
    directory_list = os.listdir(file_path)
    
    for directory in directory_list:
        if not os.path.isdir(os.path.join(file_path, directory)):
            directory_list.remove(directory)
            
        for file in os.listdir(os.path.join(file_path, directory)):
            directory_rename = directory.replace(' - ', '-').replace(' ', '_')
            file_rename = f'{directory_rename}_{file}'
            os.rename(os.path.join(file_path, directory, file), os.path.join(file_path, directory, file_rename))
            

def move_files_to_directories(source_file_path, destination_file_path):
    directory_list = os.listdir(source_file_path)
    
    for directory in directory_list:
        if not os.path.isdir(os.path.join(source_file_path, directory)):
            directory_list.remove(directory)
            
        for file in os.listdir(os.path.join(source_file_path, directory)):
            if file.endswith('vocals.wav') or file.endswith('mixture.wav'):
                if not os.path.exists(os.path.join(destination_file_path, directory)):
                    os.makedirs(os.path.join(destination_file_path, directory))
                
                shutil.copy(os.path.join(source_file_path, directory, file), os.path.join(destination_file_path, directory, file))

def load_audio_files(file_path):
    audio_files = {}

    for i, file in enumerate(os.listdir(file_path)):
        if file.endswith('.wav'):
            y, sr = datasets.load_audio(os.path.join(file_path, file))
            audio_files[i] = {'y': y, 'sr': sr, 'file_name': file}

    return audio_files

def convert_audio_files_to_spectrum(audio_files, destination_path):
    for i, audio_data in audio_files.items():
        y = audio_data['y']
        sr = audio_data['sr']
        file_name = audio_data['file_name'].replace('.wav', '')
        
        spectrum_data = datasets.get_all_features(y, sr, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=13)   
        taxonomy_spectrum =  list(spectrum_data.keys())
        
        for spectrum in taxonomy_spectrum:
            if not os.path.isdir(os.path.join(destination_path, spectrum)):
                os.makedirs(os.path.join(destination_path, spectrum))
            np.save(os.path.join(destination_path, spectrum, f'{i}_{file_name}_{spectrum}'), spectrum_data[spectrum])
        
        frames = ['spectrogram_frame', 'mel_spectrogram_frame']
        for frame in frames:
            if not os.path.isdir(os.path.join(destination_path, frame)):
                os.makedirs(os.path.join(destination_path, frame))
            
            if frame == 'spectrogram_frame':
                spectrogram_frame = datasets.get_spectrogram_frames(y, sr, n_fft=2048, hop_length=512)
                np.save(os.path.join(destination_path, frame, f'{i}_{file_name}_{frame}'), spectrogram_frame)    
            else:
                mel_spectrogram_frame = datasets.get_mel_spectrogram_frames(y, sr, n_fft=2048, hop_length=512, n_mels=128)    
                np.save(os.path.join(destination_path, frame, f'{i}_{file_name}_{frame}'), mel_spectrogram_frame)
        
import os
import re


def extract_title(filename, is_original):
    if is_original:
        return re.sub(r'^\d+_(.+?)_vocals.*$', r'\1', filename)
    else:
        return re.sub(r'^\d+_\d+_(.+?)_mixture.*$', r'\1', filename)

def rename_sep_files(original_dir, sep_dir):
    for subdir in os.listdir(original_dir):
        original_subdir = os.path.join(original_dir, subdir)
        sep_subdir = os.path.join(sep_dir, subdir)
        
        if not os.path.isdir(original_subdir) or not os.path.exists(sep_subdir):
            continue

        print(f"Processing: {subdir}")
        
        original_files = [f for f in os.listdir(original_subdir) if f.endswith('.npy')]

        for sep_file in os.listdir(sep_subdir):
            if sep_file.endswith('.npy'):
                sep_title = extract_title(sep_file, False)
                for orig_file in original_files:
                    extract_orig_title = extract_title(orig_file, True)
                    if sep_title == extract_orig_title:
                        old_path = os.path.join(sep_subdir, sep_file)
                        new_path = os.path.join(sep_subdir, orig_file)
                        os.rename(old_path, new_path)
                        print(f"Renamed {sep_file} to {orig_file}")
                        break
        
                    
result_path_train = r'C:\data\Music\musdb18hq\result\train\original'
result_path_test = r'C:\data\Music\musdb18hq\result\test\original'

result_sep_path_train = r'C:\data\Music\musdb18hq\result\train\sep'
result_sep_path_test = r'C:\data\Music\musdb18hq\result\test\sep'

result_convert_path_train = r'C:\data\Music\musdb18hq\result\train\convert'
result_convert_path_test = r'C:\data\Music\musdb18hq\result\test\convert'


original_path_train = r'C:\data\Music\musdb18hq\vocals\train'
original_path_test = r'C:\data\Music\musdb18hq\vocals\test'

sep_path_train = r'C:\data\Music\musdb18hq\mixture\train\BS-Roformer-1297'
sep_path_test = r'C:\data\Music\musdb18hq\mixture\test\BS-Roformer-1297'

convert_roformer_path_train = r'C:\data\Music\musdb18hq\mixture\train\BS-Roformer-1297\convert'
convert_roformer_path_test = r'C:\data\Music\musdb18hq\mixture\test\BS-Roformer-1297\convert'
            
# original_audio_train = load_audio_files(original_path_train)            
# original_audio_test = load_audio_files(original_path_test)
            
# convert_audio_files_to_spectrum(original_audio_train, result_path_train)
# convert_audio_files_to_spectrum(original_audio_test, result_path_test)

# print("===================================== End of the Original data ======================================")

# sep_audio_train = load_audio_files(sep_path_train)
# sep_audio_test = load_audio_files(sep_path_test)

# convert_audio_files_to_spectrum(sep_audio_train, result_sep_path_train)
# convert_audio_files_to_spectrum(sep_audio_test, result_sep_path_test)

# print("===================================== End of the Separated data ======================================")

# convert_audio_train = load_audio_files(convert_roformer_path_train)
# convert_audio_test = load_audio_files(convert_roformer_path_test)

# convert_audio_files_to_spectrum(convert_audio_train, result_convert_path_train)
# convert_audio_files_to_spectrum(convert_audio_test, result_convert_path_test)


# # Usage
original_dir = r'C:\data\Music\musdb18hq\result\train\original'
# sep_dir = r'C:\data\Music\musdb18hq\result\train\sep'
convert_dir = r'C:\data\Music\musdb18hq\result\train\convert'
# rename_sep_files(original_dir, sep_dir)
rename_sep_files(original_dir, convert_dir)