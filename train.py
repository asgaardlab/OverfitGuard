"""
Training overfitting detectors
"""
import time
import pathlib
import argparse
from datetime import datetime

import sklearn
from sklearn.metrics import classification_report

from core import helper
from core import dataloader
from core import corrbased
from core import tsprocess
from core import tsclassifiers


timer = time.perf_counter

MODEL_MAP = {
    "spearman": corrbased.SpearmanDetector,
    "pearson": corrbased.PearsonDetector,
    "autocorr": corrbased.AutocorrDetector,
    "knndtw": tsclassifiers.KNNDTW,
    "hmmgmm": tsclassifiers.HMMGMMClassifier,
    "tsf": tsclassifiers.TimeSeriesForest,
    "tsbf": tsclassifiers.TimeSeriesBagOfFeatures,
    "saxvsm": tsclassifiers.SymbolicAggreApproxVecSpace,
    "bossvs": tsclassifiers.BagOfSFASymbolsVecSpace,
}

CORRELATION_BASED_METHODS = [
    "spearman", "pearson", "autocorr",
]

TIME_SERIES_CLASSIFIERS = [
    "knndtw", "hmmgmm", "tsf", "tsbf", "saxvsm", "bossvs"
]


def main():
    """
    Script for training
    """
    parser = argparse.ArgumentParser(
        description='Training overfitting detectors',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'model',
        type=str,
        choices= CORRELATION_BASED_METHODS + TIME_SERIES_CLASSIFIERS,
        help='Correlation-based overfitting detection:\n'
             '  spearman - Spearman correlation metric\n'
             '  pearson  - Pearson correlation metric\n'
             '  autocorr - Autocorrelation correlation metric (time-lagged Pearson)\n'
             'Time-series classifiers for overfitting detection:\n'
             '  knndtw - K-Nearest Neighbors with Dynamic Time Warping\n'
             '  hmmgmm - Hidden Markov Model with Gaussian Mixture Model emissions\n'
             '  tsf    - Time Series Forest\n'
             '  tsbf   - Time Series Bag-of-Features\n'
             '  saxvsm - Symbolic Aggregate approXimation and Vector Space Model\n'
             '  bossvs - Bag-of-SFA Symbols in Vector Space\n',
    )
    parser.add_argument('data_path', type=str, help='Path for training logs')
    parser.add_argument('out_path', type=str, help='Path for saving models')
    parser.add_argument('stretch_length', type=int, help='stretch length for time series classifiers', default=1000, nargs='?')
    parser.add_argument('--zero_pad', help='Whether using zero padding as the preprocess method for KNN-DTW', action='store_true')
    args = parser.parse_args()
    print(args)

    training_set = dataloader.TrainingLogDataset(args.data_path)
    training_set.loadDataset()
    print(training_set)

    detector = MODEL_MAP[args.model]()
    if args.model in TIME_SERIES_CLASSIFIERS:
        if args.model == "knndtw" and args.zero_pad:
            detector.preprocessor = tsclassifiers.ZeroPadPreprocessor(args.stretch_length)
        # elif args.model == "hmmgmm":
        #     detector.preprocessor = tsclassifiers.DumbPreprocessor(args.stretch_length)
        else:
            detector.preprocessor = tsclassifiers.StretchPreprocessor(args.stretch_length)
        print("preprocessor:", detector.preprocessor)
        trainset_data = detector.preprocessor.process(training_set.data)
        # trainset_data = [d["monitor_metric"] for d in training_set.data]
        # trainset_data = list(map(tsprocess.standarize, trainset_data))
        # trainset_data = [tsprocess.stretchData(d, args.stretch_length) for d in trainset_data]
        # detector.train(trainset_data, training_set.labels)
    else:
        trainset_data = training_set.data

    print("detector:", detector)

    if hasattr(detector, "tuneParameters"):
        print("Tuning hyper parameters...")
        t0 = timer()
        cls = detector.tuneParameters(trainset_data, training_set.labels)
        t1 = timer()
        tuning_time = t1 - t0
        print(f"Tuning hyper parameters time: {tuning_time}")
        print(f"best_params: {cls.best_params_}")
        print(f"best_estimator: {cls.best_estimator_}")
        print(f"best_score_: {cls.best_score_}")
    else:
        print("do not need tuning parameters")
        tuning_time = 0.0

    print("Training...")
    t0 = timer()
    detector.train(trainset_data, training_set.labels)
    t1 = timer()
    training_time = t1 - t0
    print(f"Training time: {training_time}")
    out_path = pathlib.Path(args.out_path)
    out_path.mkdir(exist_ok=True)

    now_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    helper.savePkl(detector, out_path / f"{args.model}_{now_datetime}.pkl")

    t0 = timer()
    preds = detector.predict(trainset_data)
    t1 = timer()
    inference_time = t1 - t0
    print(f"Inference time: {inference_time}")

    report = classification_report(training_set.labels, preds, digits=3)
    print(report)
    with open(out_path / f"{args.model}_{now_datetime}_train_report.txt", "w") as f:
        f.write(f"Tuning parameter time: {tuning_time}\n")
        f.write(f"Best params: {cls.best_params_}\n")
        f.write(f"Training time: {training_time}\n")
        f.write(f"Inference time: {inference_time}\n")
        f.write(report)


if __name__ == '__main__':
    main()
