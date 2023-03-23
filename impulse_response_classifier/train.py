from impulse_response_classifier.path_collector import PathCollector
from impulse_response_classifier.data_generator import DataGenerator
from impulse_response_classifier.model import IRClassifier
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--savedir', help='Model save directory', required=False)
    parser.add_argument('--train', help='Train data directory')
    parser.add_argument('--test', help='Test data directory', required=False)
    args = parser.parse_args()

    if not args.savedir:
        args.savedir = 'model'
    return args

def main():
    arguments = parse_args()
    collector = PathCollector(arguments.train)

    if arguments.test:
        train, val, _ = collector.split_data(train_ratio=0.7, val_ratio=0.3)
        test_collector = PathCollector(arguments.test)
        test = test_collector.collect()
    else:
        train, val, test = collector.split_data(train_ratio=0.6, val_ratio=0.3)

    train_data = DataGenerator(train)
    val_data = DataGenerator(val)
    test_data = DataGenerator(test)

    model = IRClassifier(arguments.savedir)
    model.train(train_data, val_data)
    model.evaluate(test_data)

if __name__ == "__main__":
    main()