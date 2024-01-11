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


TEST_DATA_PATH = pathlib.Path("./data/testing/real_world_data")
test_set = dataloader.TrainingLogDataset(TEST_DATA_PATH)
test_set.loadDataset()

OUT_PATH = pathlib.Path("./out/test_whole_history")
OUT_PATH.mkdir(exist_ok=True)
print(test_set)

models_path = pathlib.Path("./models")
for cls_name in ["tsf", "tsbf", "bossvs", "hmmgmm", "saxvsm", "knndtw"]:
    print("="*9, cls_name, "="*9)
    model_path = list(models_path.glob(f"{cls_name}_*.pkl"))[0]
    model = helper.readPkl(model_path)

    def addInfo(classifier_stop_res):
        dst_len = len(classifier_stop_res["total_time"])
        classifier_stop_res["label"] = test_set.labels[:dst_len]
        classifier_stop_res["name"] = test_set.names[:dst_len]
        classifier_stop_res["window_size"] = [10] * dst_len
        classifier_stop_res["step"] = [10] * dst_len
        return classifier_stop_res

    classifier_stop_res = {
        "is_overfit": [],
        "total_time": [],
        "timer_count": [],
    }
    for idx, name in enumerate(test_set.names):
        idx = test_set.names.index(name)
        cur_data = test_set.data[idx]
        total_time = 0
        timer_count = 0
        is_overfit = []
        for i in range(len(cur_data["monitor_metric"])):
            if i < 10:
                is_overfit.append(0)
                continue
            end_epoch = i
            window_data = {n: d[:end_epoch] for n, d in cur_data.items()}
            if hasattr(model, "preprocessor"):
                processed_data = model.preprocessor.process([window_data])
            else:
                processed_data = [window_data]
            t1 = timer()
            res = model.predict(processed_data)
            t2 = timer()
            total_time += t2 - t1
            timer_count += 1
            is_overfit.append(int(res[0]))
        classifier_stop_res["is_overfit"].append(is_overfit)
        classifier_stop_res["total_time"].append(total_time)
        classifier_stop_res["timer_count"].append(timer_count)
        # break
        if idx % 50 == 0:
            print(f"{idx}/{len(test_set.names)}")
            classifier_stop_res = addInfo(classifier_stop_res)
            tmp = pd.DataFrame.from_dict(classifier_stop_res)
            tmp.to_csv(OUT_PATH / f"{model_path.stem}.csv", index=False)
    classifier_stop_res = addInfo(classifier_stop_res)
    classifier_stop_res = pd.DataFrame.from_dict(classifier_stop_res)
    classifier_stop_res.to_csv(OUT_PATH / f"{model_path.stem}.csv", index=False)
