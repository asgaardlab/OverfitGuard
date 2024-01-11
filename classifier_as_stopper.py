import pathlib

import numpy as np
import pandas as pd

from core import dataloader
from core import helper
import sklearn
from sklearn.metrics import classification_report
import argparse
import time
timer = time.perf_counter

parser = argparse.ArgumentParser(description='time series classifiers as stopper.')
parser.add_argument(
    'window_size', type=int, help='window size for classifers'
)
parser.add_argument(
    'step_size', type=int, help='step size for classifers'
)
args = parser.parse_args()
print(args)

TRAIN_DATA_PATH = pathlib.Path("./data/testing/real_world_data")
training_set = dataloader.TrainingLogDataset(TRAIN_DATA_PATH)
training_set.loadDataset()

OUT_PATH = pathlib.Path("./out/test_cmp_early_stop_step10")
OUT_PATH.mkdir(exist_ok=True)
print(training_set)

models_path = pathlib.Path("./models")
for cls_name in ["tsf", "tsbf", "bossvs", "hmmgmm", "saxvsm", "knndtw"]:
    print("="*9, cls_name, "="*9)
    model_path = list(models_path.glob(f"{cls_name}_*.pkl"))[0]
    model = helper.readPkl(model_path)

    classifier_window = args.window_size
    step = args.step_size

    def addInfo(classifier_stop_res):
        dst_len = len(classifier_stop_res["is_stopped"])
        classifier_stop_res["label"] = training_set.labels[:dst_len]
        classifier_stop_res["name"] = training_set.names[:dst_len]
        classifier_stop_res["window_size"] = [classifier_window] * dst_len
        classifier_stop_res["step"] = [step] * dst_len
        return classifier_stop_res

    classifier_stop_res = {
        "is_stopped": [],
        "stop_epoch": [],
        "best_epoch": [],
        "best_loss": [],
        "total_time": [],
        "timer_count": [],
    }
    for idx, name in enumerate(training_set.names):
        idx = training_set.names.index(name)
        cur_data = training_set.data[idx]
        total_time = 0
        timer_count = 0
        for i in range(0, len(cur_data["monitor_metric"]) - classifier_window + step, step):
            end_epoch = i + classifier_window
            window_data = {n: d[i:end_epoch] for n, d in cur_data.items()}
            processed_data = model.preprocessor.process([window_data])
            t1 = timer()
            res = model.predict(processed_data)
            t2 = timer()
            total_time += t2 - t1
            timer_count += 1
            if res:
                best_epoch = np.argmin(cur_data["monitor_metric"][:end_epoch])
                best_loss = cur_data["monitor_metric"][best_epoch]
                classifier_stop_res["is_stopped"].append(1)
                classifier_stop_res["stop_epoch"].append(end_epoch - 1)
                break
        else:
            best_epoch = np.argmin(cur_data["monitor_metric"])
            best_loss = cur_data["monitor_metric"][best_epoch]
            classifier_stop_res["is_stopped"].append(0)
            classifier_stop_res["stop_epoch"].append(len(cur_data["monitor_metric"]) - 1)
        classifier_stop_res["best_epoch"].append(best_epoch)
        classifier_stop_res["best_loss"].append(best_loss)
        classifier_stop_res["total_time"].append(total_time)
        classifier_stop_res["timer_count"].append(timer_count)
        # break
        if idx % 50 == 0:
            print(f"{idx}/{len(training_set.names)}")
            classifier_stop_res = addInfo(classifier_stop_res)
            tmp = pd.DataFrame.from_dict(classifier_stop_res)
            tmp.to_csv(OUT_PATH / f"{model_path.stem}_{classifier_window}_{step}.csv", index=False)
    classifier_stop_res = addInfo(classifier_stop_res)
    classifier_stop_res = pd.DataFrame.from_dict(classifier_stop_res)
    classifier_stop_res.to_csv(OUT_PATH / f"{model_path.stem}_{classifier_window}_{step}.csv", index=False)
