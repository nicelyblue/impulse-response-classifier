from impulse_response_classifier.path_collector import PathCollector
from impulse_response_classifier.data_generator import DataGenerator
from impulse_response_classifier.model import IRClassifier
import os

if __name__ == "__main__":
    collector = PathCollector(os.path.relpath('data'), ['.wav', '.Wav'])
    train, val, test = collector.split_data()

    train_data = DataGenerator(train)
    val_data = DataGenerator(val)
    test_data = DataGenerator(test)

    model = IRClassifier([(15,), (1025, 513, 1)])
    model.train(train_data=train_data, val_data=val_data, epochs=100)
    model.evaluate(test_data=test_data)