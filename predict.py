"""
Training overfitting detectors
"""
import time
import argparse
import pathlib
from datetime import datetime

import sklearn
from sklearn.metrics import classification_report

from core import helper
from core import dataloader
from core import corrbased


timer = time.perf_counter


def main():
    """
    Script for training
    """
    parser = argparse.ArgumentParser(
        description='Predicting overfitting',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('model_path', type=str, help='Path for the saved model')
    parser.add_argument(
        'data_path', type=str,
        help='Path for training logs that require overfitting detection'
    )
    parser.add_argument('out_path', type=str, help='Path for saving results')
    parser.add_argument('--with_label', action='store_true')
    args = parser.parse_args()
    print(args)

    test_set = dataloader.TrainingLogDataset(args.data_path, withoutLabel=not args.with_label)
    test_set.loadDataset()
    print(test_set)

    model_path = pathlib.Path(args.model_path)
    detector = helper.readPkl(model_path)
    out_path = pathlib.Path(args.out_path)
    out_path.mkdir(exist_ok=True)

    if hasattr(detector, "preprocessor") and detector.preprocessor is not None:
        test_data = detector.preprocessor.process(test_set.data)
    else:
        test_data = test_set.data

    t0 = timer()
    pred_res = detector.predict(test_data)
    t1 = timer()
    inference_time = t1 - t0
    print(f"Inference time: {inference_time}")

    pred_res_dict = {}
    for name, res in zip(test_set.names, pred_res):
        pred_res_dict[name] = int(res)
    helper.saveJson(pred_res_dict, out_path / f"{model_path.stem}_pred_res.json")
    if test_set.labels:
        report = classification_report(test_set.labels, pred_res, digits=3)
        print(report)
        with open(out_path / f"{model_path.stem}_pred_report.txt", "w") as f:
            f.write(f"Inference time: {inference_time}\n")
            f.write(report)


if __name__ == '__main__':
    main()
