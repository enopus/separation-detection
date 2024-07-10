import os
import shutil
import soundfile as sf
import librosa
import numpy as np

import separationDetection.data.datasets as datasets

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

original_path_train = r'C:\data\Music\musdb18hq\train'
original_path_test = r'C:\data\Music\musdb18hq\test'
destination_path_train = r'C:\data\Music\musdb18hq\vocals'
destination_path_train = r'C:\data\Music\musdb18hq\mixture'
            
rename_file(original_path_train)
rename_file(original_path_test)
            
