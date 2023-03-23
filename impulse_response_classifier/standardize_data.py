from pydub import AudioSegment
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Standardize the data.')
    parser.add_argument('--data', nargs='+', help='Data directories.')
    args = parser.parse_args()
    return args

def find_wav_files(path):
    wav_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".wav") or file.endswith(".Wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files

def get_duration(file_path):
    audio = AudioSegment.from_wav(file_path)
    return audio.duration_seconds

def pad_wav_file(file_path, max_duration):
    audio = AudioSegment.from_wav(file_path)
    duration = audio.duration_seconds

    if duration == max_duration:
        return

    padding_duration = max_duration - duration
    silence = AudioSegment.silent(duration=padding_duration * 1000)
    padded_audio = audio + silence
    padded_audio.export(file_path, format="wav")

def main():
    arguments = parse_args()
    
    for path in arguments.data:
        wav_files = find_wav_files(path)
        max_duration = max(get_duration(file) for file in wav_files)

        for wav_file in wav_files:
            pad_wav_file(wav_file, max_duration)

if __name__ == "__main__":
    main()
