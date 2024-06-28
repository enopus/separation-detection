import os
import pandas as pd
import librosa
import numpy as np
import random
import soundfile as sf

def load_erroneous_files(file_path):
    df = pd.read_csv(file_path, header=None)  # Assuming there's no header
    erroneous_files = df[0].tolist()  # Assuming filenames are in the first column
    return erroneous_files

def delete_files(files_to_delete, directory):
    for filename in files_to_delete:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print(f"File not found: {file_path}")

def load_audio_files(directory, excluded_files):
    audio_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav") and filename not in excluded_files:
            file_path = os.path.join(directory, filename)
            try:
                y, sr = librosa.load(file_path, sr=None)
                audio_files.append((y, sr))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return audio_files

def add_silence(duration, sr):
    return np.zeros(int(duration * sr))

def concatenate_audios(audio_files):
    concatenated_audio = np.array([])
    total_length = 0
    sr = audio_files[0][1]

    while total_length <= 23:
        for y, sr in audio_files:
            silence_duration = random.uniform(0.2, 0.5)
            silence = add_silence(silence_duration, sr)
            concatenated_audio = np.concatenate((concatenated_audio, y, silence))
            total_length = librosa.get_duration(y=concatenated_audio, sr=sr)
            if total_length > 23:
                break

    return concatenated_audio, sr

def main():
    error_file_path = "path_to_error_list.csv"  # Path to the CSV file containing filenames of erroneous files
    audio_directory = "path_to_your_audio_files_directory"  # Path to the directory containing audio files

    # Load erroneous files list
    erroneous_files = load_erroneous_files(error_file_path)
    
    # Delete erroneous files
    delete_files(erroneous_files, audio_directory)

    # Load remaining audio files
    audio_files = load_audio_files(audio_directory, erroneous_files)
    
    # Concatenate audio files
    concatenated_audio, sr = concatenate_audios(audio_files)

    output_file = "concatenated_output.wav"
    sf.write(output_file, concatenated_audio, sr)
    print(f"Output saved to {output_file}, total length: {librosa.get_duration(y=concatenated_audio, sr=sr)} seconds")

if __name__ == "__main__":
    main()
