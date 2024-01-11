import pathlib

import numpy as np
import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt

from core import dataloader
from core import earlystop
from core import helper
import sklearn
from sklearn.metrics import classification_report


data_dir = pathlib.Path("./out/test_whole_history")
save_dir = pathlib.Path("./out/test_whole_history_trans")
save_dir.mkdir(exist_ok=True)

TRAIN_DATA_PATH = pathlib.Path("./data/testing/real_world_data")
training_set = dataloader.TrainingLogDataset(TRAIN_DATA_PATH)
training_set.loadDataset()

STEP = 20


def addInfo(classifier_stop_res):
    dst_len = len(classifier_stop_res["best_epoch"])
    # classifier_stop_res["label"] = training_set.labels[:dst_len]
    # classifier_stop_res["name"] = training_set.names[:dst_len]
    classifier_stop_res["window_size"] = [STEP] * dst_len
    classifier_stop_res["step"] = [STEP] * dst_len
    return classifier_stop_res


for dp in data_dir.iterdir():
    classifier_stop_res = {
        # "is_stopped": [],
        # "stop_epoch": [],
        "best_epoch": [],
        "best_loss": [],
        # "total_time": [],
        # "timer_count": [],
    }
    data = pd.read_csv(dp)
    data = data.sort_values("name").reset_index()
    data["is_overfit"] = data["is_overfit"].apply(eval)
    data["is_overfit"] = data["is_overfit"].apply(np.array)
    def findStopEpoch(x):
        found_idx = np.where(x[STEP:] == 1)[0]
        if len(found_idx):
            return found_idx[0] + STEP
        return -1
    data["stop_epoch"] = data["is_overfit"].apply(findStopEpoch)
    data["is_stopped"] = data["stop_epoch"] != -1

    max_length = []
    for name, stop in zip(data["name"], data["stop_epoch"]):
        idx = training_set.names.index(name)
        cur_data = training_set.data[idx]
        if stop == -1:
            best_epoch = np.argmin(cur_data["monitor_metric"][:-1])
        else:
            best_epoch = np.argmin(cur_data["monitor_metric"][:stop + 1])
        best_loss = cur_data["monitor_metric"][best_epoch]
        classifier_stop_res["best_epoch"].append(best_epoch)
        classifier_stop_res["best_loss"].append(best_loss)
        max_length.append(len(cur_data["monitor_metric"]))
    data["max_len"] = max_length
    data.loc[~data["is_stopped"], "stop_epoch"] = data.loc[~data["is_stopped"], "max_len"] - 1

    classifier_stop_res = addInfo(classifier_stop_res)
    classifier_stop_res = pd.DataFrame.from_dict(classifier_stop_res)
    classifier_stop_res["total_time"] = data["total_time"]
    classifier_stop_res["timer_count"] = data["timer_count"]
    classifier_stop_res["is_stopped"] = data["is_stopped"]
    classifier_stop_res["stop_epoch"] = data["stop_epoch"]
    classifier_stop_res["name"] = data["name"]
    classifier_stop_res["label"] = data["label"]
    classifier_stop_res.to_csv(save_dir / f"{dp.stem}_{STEP}_{STEP}.csv", index=False)
