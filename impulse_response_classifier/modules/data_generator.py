import numpy as np
import librosa
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

class DataGenerator(Sequence):
    def __init__(self, file_paths, batch_size=32, n_classes=2):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.scaler = StandardScaler()

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_file_paths = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels()[idx * self.batch_size:(idx + 1) * self.batch_size]

        acoustic_features = []
        spectrograms = []
        for file_path in batch_file_paths:
            ac_features, spectrogram = self.extract_features(file_path)
            acoustic_features.append(ac_features)
            spectrograms.append(spectrogram)

        acoustic_features = self.scaler.fit_transform(acoustic_features)
        spectrograms = np.array(spectrograms)

        y = np.array(batch_labels)
        y = to_categorical(y, num_classes=self.n_classes)

        return [acoustic_features, spectrograms], y

    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, sr=16000, mono=True, duration=2.5)
        features = []

        features.append(librosa.feature.spectral_centroid(y=y, sr=sr))
        features.append(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features.append(librosa.feature.spectral_flatness(y=y))
        features.append(librosa.feature.spectral_contrast(y=y, sr=sr))

        features.append(librosa.feature.zero_crossing_rate(y))
        features.append(librosa.feature.rms(y=y))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.append(mfcc)

        y_early = y[:int(sr * 0.05)]
        y_late = y[int(sr * 0.05):]

        early_energy = np.sum(y_early ** 2)
        late_energy = np.sum(y_late ** 2)

        features.append(np.array([early_energy / late_energy if late_energy != 0 else 0]))

        acoustic_features = np.concatenate([feature.flatten() for feature in features])

        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram = np.expand_dims(spectrogram, axis=-1)

        return acoustic_features, spectrogram

    def labels(self):
        labels = []
        for filepath in self.file_paths:
            labels.append(int(os.path.basename(os.path.dirname(filepath))))
        return np.array(labels)
