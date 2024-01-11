import sklearn
import sklearn.metrics
import numpy as np
from scipy.stats import spearmanr, pearsonr


def autoCorr(train, val, shift):
    correlation, pvalue = pearsonr(train[shift:], val[:-shift])
    return correlation, pvalue


class CorrelationBasedDetector(object):
    def __init__(self):
        self._threshold = None

    def _calCorr(self, trainMetric, valMetric):
        raise NotImplemented

    def _evalPredRes(self, preds, labels):
        _, _, f1_scores, _ = sklearn.metrics.precision_recall_fscore_support(
            labels, preds, zero_division=0
        )
        return np.mean(f1_scores)

    def train(self, data, labels, start=-1, end=1, step=0.1):
        threshold_values = np.arange(start, end + step, step)
        results = {}
        for threshold in threshold_values:
            preds = self.predict(data, threshold)
            results[threshold] = self._evalPredRes(preds, labels)
        best_threshold = max(results, key=results.get)
        self._threshold = best_threshold
        return best_threshold

    def _calRollingCorr(self, trainMetric, monitorMetric):
        assert len(trainMetric) == len(monitorMetric)
        return [self._calCorr(trainMetric[:i+1], monitorMetric[:i+1]) for i in range(len(trainMetric))]

    def predict(self, data, threshold=None, rolling=False):
        if threshold is None:
            use_threshold = self._threshold
        else:
            use_threshold = threshold
        if rolling:
            res = np.array([
                (self._calRollingCorr(d["train_metric"], d["monitor_metric"]) < use_threshold).any()
                for d in data
            ])
        else:
            res = np.array([self._calCorr(d["train_metric"], d["monitor_metric"]) for d in data]) < use_threshold
        return res


class SpearmanDetector(CorrelationBasedDetector):
    def _calCorr(self, trainMetric, valMetric):
        if len(trainMetric) < 1:
            return np.nan
        return spearmanr(trainMetric, valMetric)[0]


class PearsonDetector(CorrelationBasedDetector):
    def _calCorr(self, trainMetric, valMetric):
        if len(trainMetric) < 1:
            return np.nan
        return pearsonr(trainMetric, valMetric)[0]


class AutocorrDetector(CorrelationBasedDetector):
    def __init__(self, shift=5):
        super(AutocorrDetector, self).__init__()
        self._shift = shift

    def _calCorr(self, trainMetric, valMetric):
        if len(trainMetric) < self._shift + 1:
            return np.nan
        return autoCorr(trainMetric, valMetric, self._shift)[0]
