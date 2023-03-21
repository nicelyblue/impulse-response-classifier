import os
import random

class PathCollector:
    def __init__(self, root_dir, extensions=['.wav', '.Wav']):
        self.root_dir = root_dir
        self.extensions = extensions
        self.file_paths = []

    def collect(self):
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in self.extensions):
                    self.file_paths.append(os.path.join(dirpath, filename))

        return self.file_paths

    def split_data(self, train_ratio=0.7, val_ratio=0.2, shuffle=True):
        file_paths = self.collect()

        if shuffle:
            random.shuffle(file_paths)

        num_samples = len(file_paths)
        num_train = int(num_samples * train_ratio)
        num_val = int(num_samples * val_ratio)

        train_paths = file_paths[:num_train]
        val_paths = file_paths[num_train:num_train+num_val]
        test_paths = file_paths[num_train+num_val:]

        return train_paths, val_paths, test_paths
