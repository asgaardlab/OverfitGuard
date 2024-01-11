import pathlib

import numpy as np
import pandas as pd

from . import helper


class TrainingLogDataset(object):
    def __init__(self, datasetPath, withoutLabel=False):
        self._dataset_path = pathlib.Path(datasetPath)
        self._overfit_data = {}
        self._non_overfit_data = {}
        self.without_label = withoutLabel
        self.names = None
        self.data = None
        self.labels = None

    def _loadDir(self, dirPath):
        assert dirPath.exists(), f"{dirPath} not exists!"
        data = {}
        for fp in dirPath.rglob("*.json"):
            data[fp.name] = loadData(fp)
        return data

    def loadDataset(self):
        assert self._dataset_path.exists(), f"{self._dataset_path} not exists!"
        if self.without_label:
            data = self._loadDir(self._dataset_path)
            self.names = list(data.keys())
            self.data = list(data.values())
        else:
            self._overfit_data = self._loadDir(self._dataset_path / "overfit")
            self._non_overfit_data = self._loadDir(self._dataset_path / "non_overfit")
            self.names = list(self._overfit_data.keys()) + list(self._non_overfit_data.keys())
            self.data = list(self._overfit_data.values()) + list(self._non_overfit_data.values())
            self.labels = [1] * len(self._overfit_data) + [0] * len(self._non_overfit_data)

    def __str__(self):
        if self.without_label:
            return f"Loaded dataset from {self._dataset_path.absolute()}:\n" \
                   f"    {len(self.data)} data, no labels \n"
        else:
            return f"Loaded dataset from {self._dataset_path.absolute()}:\n" \
                   f"    {len(self.data)} data, {len(self.labels)} labels \n" \
                   f"    {len(self._overfit_data)} overfitting samples\n" \
                   f"    {len(self._non_overfit_data)} non_overfitting samples\n"


def loadData(logPath):
    data_point = {}
    data = helper.loadJson(logPath)
    data_point["train_metric"] = np.array(data[data["train_metric"]])
    data_point["monitor_metric"] = np.array(data[data["monitor_metric"]])
    return data_point
