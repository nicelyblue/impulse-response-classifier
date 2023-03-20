import os
import numpy as np
import scipy.signal as signal
import librosa


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, filepaths, batch_size=32, shuffle=True, n_fft=1024, hop_length=512, max_len=None):
        self.filepaths = filepaths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_len = max_len
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.filepaths) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_filepaths = self.filepaths[index *
                                         self.batch_size:(index + 1) * self.batch_size]
        batch_X, batch_y = self.__data_generation(batch_filepaths)
        return batch_X, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.filepaths)

    def __data_generation(self, batch_filepaths):
        X = []
        y = []
        max_len = 0
        for filepath in batch_filepaths:
            # Load the audio signal from the .wav file
            signal, sr = librosa.load(filepath, sr=None, mono=True)

            # Calculate the time-domain features
            mean = np.mean(signal)
            var = np.var(signal)
            skewness = np.mean((signal - mean)**3) / (var**(3/2))
            kurtosis = np.mean((signal - mean)**4) / (var**2)

            # Calculate the frequency-domain features
            spectrum = np.abs(np.fft.fft(signal))
            freqs = np.fft.fftfreq(len(signal), d=1/sr)
            freqs = freqs[:len(freqs)//2]
            spectrum = spectrum[:len(spectrum)//2]
            centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
            spread = np.sqrt(np.sum(((freqs - centroid)**2)
                             * spectrum) / np.sum(spectrum))
            flatness = np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum)
            rolloff = np.interp(0.85, np.cumsum(
                spectrum)/np.sum(spectrum), freqs)

            # Calculate the time-frequency features
            f, t, spec = signal.stft(signal, fs=sr, nperseg=self.n_fft,
                                     noverlap=self.n_fft - self.hop_length, window='hamming')
            peak_freq = f[np.argmax(np.abs(spec), axis=0)]
            peak_mag = np.max(np.abs(spec), axis=0)
            bandwidth = np.sum(
                np.abs(spec) > 0.5 * np.max(np.abs(spec), axis=0), axis=0) * (f[1] - f[0])
            tf_entropy = np.sum(spec*np.log(np.abs(spec))
                                ) / np.sum(np.abs(spec))

            # Calculate the echo features
            ir = signal / np.max(np.abs(signal))
            ir_energy = np.sum(ir**2)
            ir_decay = -60
            ir_decay_rate = 0.0
            for i in range(0, len(ir)):
                if ir[i]**2 < ir_energy * 0.05:
                    ir_decay = i
                    break
            if ir_decay > 0:
                ir_decay_rate = -60.0 / (ir_decay * sr)
            ir_peak = np.max(np.abs(ir))
            ir_peak_time = np.argmax(np.abs(ir))

            # Calculate the spectrogram
            S = np.abs(librosa.stft(signal, n_fft=self.n_fft,
                       hop_length=self.hop_length))
            if self.max_len is None:
                max_len = S.shape[1]
            else:
                max_len = self.max_len

            # Pad the spectrogram to the maximum length
            S_padded = np.zeros((S.shape[0], max_len))
            S_padded[:, :S.shape[1]] = S[:, :max_len]

            # Concatenate all the features
            features = np.concatenate([
                np.array([mean, var, skewness, kurtosis]),
                np.array([centroid, spread, flatness, rolloff]),
                np.array([np.mean(peak_freq), np.mean(peak_mag),
                         np.mean(bandwidth), tf_entropy]),
                np.array([ir_energy, ir_decay_rate, ir_peak, ir_peak_time])
            ])

            # Add the spectrogram and features to the batch
            X.append(np.transpose(S_padded))
            y.append(int(os.path.basename(os.path.dirname(filepath))))

        # Convert the batch to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Convert the class labels to binary vectors
        y = tf.keras.utils.to_categorical(y, num_classes=2)

        return X, y
