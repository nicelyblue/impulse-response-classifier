import os
import numpy as np
import scipy.signal as signal
import librosa
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, filepaths, batch_size=32, shuffle=True, n_fft=2048, hop_length=1024, max_len=1024):
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
        x_1, x_2, y = self.__data_generation(batch_filepaths)
        x_2 = DataGenerator.normalize_spectrogram(x_2)
        return [x_1, x_2], y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.filepaths)     

    @staticmethod
    def normalize_spectrogram(data, axis=(0, 1)):
        # Squeeze the last dimension with size 1
        data = np.squeeze(data)

        # Compute the mean and standard deviation along the desired axis
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)

        # Broadcast the mean and standard deviation back to the original shape
        mean_broadcast = np.broadcast_to(mean, data.shape)
        std_broadcast = np.broadcast_to(std, data.shape)

        # Normalize the data
        data = (data - mean_broadcast) / std_broadcast

        # Add the last dimension back to the normalized data
        data = np.expand_dims(data, axis=-1)

        return data

    def __data_generation(self, batch_filepaths):
        x_1 = []
        x_2 = []
        y = []
        max_len = 0
        for filepath in batch_filepaths:
            sig, sr = librosa.load(filepath, sr=None, mono=True)

            mean = np.mean(sig)
            var = np.var(sig)
            skewness = np.mean((sig - mean)**3) / (var**(3/2))
            kurtosis = np.mean((sig - mean)**4) / (var**2)

            spectrum = np.abs(np.fft.fft(sig))
            freqs = np.fft.fftfreq(len(sig), d=1/sr)
            freqs = freqs[:len(freqs)//2]
            spectrum = spectrum[:len(spectrum)//2]
            centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
            spread = np.sqrt(np.sum(((freqs - centroid)**2)
                             * spectrum) / np.sum(spectrum))
            flatness = np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum)
            rolloff = np.interp(0.85, np.cumsum(
                spectrum)/np.sum(spectrum), freqs)

            f, t, spec = signal.stft(sig, fs=sr, nperseg=self.n_fft,
                                     noverlap=self.n_fft - self.hop_length, window='hamming')
            peak_freq = f[np.argmax(np.abs(spec), axis=0)]
            peak_mag = np.max(np.abs(spec), axis=0)
            bandwidth = np.sum(
                np.abs(spec) > 0.5 * np.max(np.abs(spec), axis=0), axis=0) * (f[1] - f[0])

            ir = sig / np.max(np.abs(sig))
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

            S = np.abs(librosa.stft(sig, n_fft=self.n_fft,
                       hop_length=self.hop_length))
            if self.max_len is None:
                max_len = S.shape[1]
            else:
                max_len = self.max_len

            S_padded = np.zeros((S.shape[0], max_len))
            S_padded[:, :S.shape[1]] = S[:, :max_len]

            features = np.concatenate([
                np.array([mean, var, skewness, kurtosis]),
                np.array([centroid, spread, flatness, rolloff]),
                np.array([np.mean(peak_freq), np.mean(peak_mag),
                         np.mean(bandwidth)]),
                np.array([ir_energy, ir_decay_rate, ir_peak, ir_peak_time])
            ])

            x_1.append(features)
            x_2.append(np.transpose(S_padded))
            y.append(int(os.path.basename(os.path.dirname(filepath))))

        y = np.array(y)
        y = tf.keras.utils.to_categorical(y, num_classes=2)

        return np.asarray(x_1).astype('float32'), np.asarray(x_2).astype('float32'), y
    
    def labels(self):
        labels = []
        for filepath in self.filepaths:
            labels.append(int(os.path.basename(os.path.dirname(filepath))))
        return np.array(labels)