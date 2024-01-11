# Replication package of the paper "Keeping Deep Learning Models in Check: A History-Based Approach to Mitigate Overfitting"

This repository provides all the data, code, notebook, and the trained models required to replicate our paper:

- The datasets can be found under the [./data](./data) folder, where:
  - the [training](./data/training) folder contains the simulated dataset
  - the [testing](./data/testing) folder contains the real-world dataset
- Using [./train.py](./train.py) for training and [./predict.py](./predict.py) for prediction
- Using the notebook [./reproduce.ipynb](./reproduce.ipynb) to see the results and figures of our paper
- The trained models can be found under the [models](./models) folder
- The full list of surveyed paper can be found in [surveyed_paper.md](surveyed_paper.md)

## Setup environment

This repository is based on Python `3.8.13` version.

### Conda

```sh
conda env create -f environment.yml
```

### pip

```sh
pip install -r requirements.txt
```

## Data preparation

This project is for detecting overfitting based on training logs.
The format of the training log should be a `json` file and contain:

- Training loss
- Validation loss

The names of the keys should be specified as `train_metric` and `monitor_metric`.
For example, a training log stores training loss under key named `training_loss`
and validation loss under key named `validation_loss`:

```json
{
  "training_loss": [0.720, 0.716, ...],
  "validation_loss": [0.707, 0.706, ...],
  "train_metric": "training_loss",
  "monitor_metric": "validation_loss"
}
```

Example training logs can be found in [./data/training/dataset_exp4](./data/training/dataset_exp4)
folder.

## Training Overfitting Detection Methods

### Correlation-based Methods

We provide three methods:

- Spearman (recommend)
- Pearson
- Autocorrelation

Training by:

```sh
python train.py spearman ./data/training/dataset_exp4 ./out
python train.py pearson ./data/training/dataset_exp4 ./out
python train.py autocorr ./data/training/dataset_exp4 ./out
```

### Time series classifiers

We provide 6 time series classifiers:

- KNN-DTW: K-Nearest Neighbors with Dynamic Time Warping (recommended)
- TSF: Time Series Forest (recommended)
- TSBF: Time Series Bag-of-Features
- HMM-GMM: Hidden Markov Model with Gaussian Mixture Model emissions
- BOSSVS: Bag-of-SFA Symbols in Vector Space
- SAX-VSM: Symbolic Aggregate approXimation and Vector Space Model

`KNN-DTW` has the highest F1-score in our experiments, but it requires more
time to compute than other methods. `TSF` has a lower F1-score than `KNN-DTW`
but faster.

```sh
python train.py knndtw ./data/training/dataset_exp4 ./out --zero_pad
python train.py tsf ./data/training/dataset_exp4 ./out
python train.py tsbf ./data/training/dataset_exp4 ./out
python train.py hmmgmm ./data/training/dataset_exp4 ./out
python train.py bossvs ./data/training/dataset_exp4 ./out
python train.py saxvsm ./data/training/dataset_exp4 ./out
```

The trained models are saved under the `./out` folder for later use.

## Using the Trained Detection Methods

### Overfitting detection

Preparing the training logs (one or more) that requires overfitting detection
and run the following code:

```sh
python predict.py TRAINED_METHOD_PATH DATA_DIR OUTPUT_DIR
# examples
python predict.py ./out/spearman_{DATE}.pkl ./data/testing/real_world_data/ ./out
python predict.py ./out/knndtw_{DATE}.pkl ./data/testing/real_world_data/ ./out
python predict.py ./out/tsf_{DATE}.pkl ./data/testing/real_world_data/ ./out
```

### Overfitting prevention

The trained model can be used for overfitting prevention:

- based on the rolling window: [./classifier_as_stopper.py](./classifier_as_stopper.py)
- based on the whole history: [./classifier_as_stopper_whole_his.py](./classifier_as_stopper_whole_his.py)
