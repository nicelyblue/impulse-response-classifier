# Impulse Response Classifier
Is your impulse response synthetic? Time to find out.

This repository contains the code for an impulse response classifier that can learn to classify whether an impulse response is simulated or real.

## Installation

To install this project, simply clone this repository and run the following command:

```bash
$ pip install -e .
```

## Usage

To standardize the dataset, run the following command:

```bash
$ standardize_data --data /path/to/data/to/standardize
```

Run the following command to train the model:

```bash
$ train --train /path/to/train/data --test /path/to/test/data --savedir /path/to/model
```

To evaluate the model, run the following command:

```bash
$ evaluate --eval /path/to/eval/data --savedir /path/to/model --resultsdir /path/to/evaluation/results
```

//  Note that the data must be organized into two subfolders: '0' and '1', representing 'simulated' and 'real' labels, respectively. The only supported format is '.wav'.

## Contributing

Contributions to this project are always welcome! To contribute, please follow these steps:

1. Fork this repository
2. Create a new branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
