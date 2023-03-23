from impulse_response_classifier.modules.path_collector import PathCollector
from impulse_response_classifier.modules.data_generator import DataGenerator
from impulse_response_classifier.modules.model import IRClassifier
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the model.')
    parser.add_argument('--savedir', help='Model save directory', required=False)
    parser.add_argument('--evaldir', help='Evaluation data directory')
    args = parser.parse_args()

    if not args.savedir:
        args.savedir = 'model'
    return args

def main():
    arguments = parse_args()
    collector = PathCollector(arguments.evaldir)
    test = collector.collect()
    test_data = DataGenerator(test)

    model = IRClassifier(arguments.savedir)
    model.evaluate(test_data=test_data)

if __name__ == "__main__":
    main()
