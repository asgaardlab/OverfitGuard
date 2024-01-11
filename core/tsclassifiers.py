import multiprocessing
from functools import partial

import pyts
import pyts.classification
import sklearn.metrics
from sequentia.classifiers import GMMHMM, HMMClassifier
from hmmlearn import hmm as HMM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

import numpy as np
from fastdtw import fastdtw

from . import tsprocess

RANDOM_STATE = 57
GRID_SEARCH_JOBS = multiprocessing.cpu_count()
MINUS_INF = -99999999


class StretchPreprocessor(object):
    def __init__(self, stretchLen):
        self._stretch_length = stretchLen

    def process(self, trainingSetData):
        data = [d["monitor_metric"] for d in trainingSetData]
        data = list(map(tsprocess.standarize, data))
        data = [np.array(tsprocess.stretchData(d, self._stretch_length)) for d in data]
        return data


class DumbPreprocessor(object):
    def __init__(self, stretchLen):
        self._stretch_length = stretchLen

    def process(self, trainingSetData):
        data = [d["monitor_metric"] for d in trainingSetData]
        print("fffff")
        # data = list(map(tsprocess.standarize, data))
        data = [np.array(tsprocess.stretchData(d, self._stretch_length)) for d in data]
        return data


class ZeroPadPreprocessor(object):
    def __init__(self, stretchLen):
        self._stretch_length = stretchLen

    def process(self, trainingSetData):
        data = [d["monitor_metric"] for d in trainingSetData]
        data = list(map(tsprocess.standarize, data))
        data = [np.array(d + [-MINUS_INF] * (self._stretch_length - len(d))) for d in data]
        return data


class TimeSeriesClassifier(object):
    def __init__(self):
        self._clf = None
        self.preprocessor = None
        self.parameters = None

    def _getHyperparameters(self):
        raise NotImplementedError

    def _createModel(self, **kwargs):
        raise NotImplementedError

    def tuneParameters(self, data, labels):
        clf = GridSearchCV(
            self._createModel(), self._getHyperparameters(), cv=3, verbose=1, n_jobs=GRID_SEARCH_JOBS,
            scoring="f1_macro"
        )
        clf.fit(data, labels)
        self.parameters = clf.best_params_
        return clf

    def train(self, data, labels):
        self._clf = self._createModel(**self.parameters)
        self._clf.fit(data, labels)

    def predict(self, data):
        return self._clf.predict(data)


def DTW(a, b):
    # filter out None
    a = list(filter(lambda x: x != -MINUS_INF, a))
    b = list(filter(lambda x: x != -MINUS_INF, b))
    distance = fastdtw(a, b)[0]
    return distance


class KNNDTW(TimeSeriesClassifier):
    def _getHyperparameters(self):
        return {'n_neighbors': [3, 5, 7, 9]}

    def _createModel(self, **kwargs):
        return KNeighborsClassifier(metric=DTW, **kwargs)

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, TransformerMixin
from sklearn.utils.estimator_checks import check_estimator


class DumbClassifier(object):
    def predict(self, data):
        return [-1 for d in data]


class MyHMMGMM(BaseEstimator):
    def __init__(self, n_states=3, n_components=5, topology="linear", random_state=RANDOM_STATE, repeat=True):
        self.n_states = n_states
        self.n_components = n_components
        self.topology = topology
        self.random_state = random_state
        self.repeat = repeat
        # pass
        # super(MyHMMGMM, self).__init__()
        # self.kwargs = kwargs
        # self.hmms = []

    def fit(self, data, labels):
        labels_set = set(labels)
        train_set = {l: [] for l in labels_set}
        for x, y in zip(data, labels):
            train_set[y].append(np.array(x))

        self.hmms = []
        try:
            for label, sequences in train_set.items():
                print(type(sequences), type(sequences[0]))
                counter = 100
                while True:
                    try:
                        print("trying...")
                        # Create a linear HMM with 3 states and 5 components in the GMM emission state distributions
                        hmm = GMMHMM(label=label, n_states=self.n_states, n_components=self.n_components, topology=self.topology, random_state=self.random_state)
                        hmm.set_random_initial()
                        hmm.set_random_transitions()

                        # Fit each HMM only on the observation sequences which had that label
                        hmm.fit(sequences)
                        break
                    except Exception as e:
                        print(e)
                        if not self.repeat:
                            raise e
                    counter = counter - 1
                    if counter == 0:
                        raise ValueError("cannot train the hmm")
                self.hmms.append(hmm)
            self.clf = HMMClassifier().fit(self.hmms)
            print("succeed!")
        except Exception as e:
            print(e)
            print("using dumb classifier")
            self.clf = DumbClassifier()
        return self
        # return super(MyHMMGMM, self).fit(self.hmms)
        # HMMClassifier().fit(self.hmms)
        # return self.model

    def predict(self, data):
        return self.clf.predict(data)


class HMMGMMClassifier(TimeSeriesClassifier):
    def _getHyperparameters(self):
        # return {
        #     'n_states': [1, 3, 5, 7],
        #     "n_components": [1, 3, 5, 7],
        #     'topology': ['ergodic', 'left-right', 'linear'],
        # }
        return {
            'n_states': [3, 5, 7, 9],
            "n_components": [3, 5, 7, 9],
            'topology': ['ergodic', 'left-right', 'linear'],
            # 'n_states': [3, ],
            # "n_components": [5, ],
            # 'topology': ['linear', ],
            # 'random_state': list(range(0, 123, 3)),
            # 'random_state': [93, ],
        }
        # return {
        #     'n_states': [3, ],
        #     "n_components": [5, ],
        #     'topology': ['linear', ],
        #     'random_state': [4440, 4440, 4440, 4440, 4440],
        # }

    def _createModel(self, **kwargs):
        return MyHMMGMM(**kwargs)

    def __init__(self):
        super(HMMGMMClassifier, self).__init__()
        self._hmm = None
        self._hmms = []
        self._labels = []

    def tuneParameters(self, data, labels):
        clf = GridSearchCV(
            self._createModel(), self._getHyperparameters(), cv=3, verbose=1, n_jobs=GRID_SEARCH_JOBS,
            scoring="f1_macro"
        )
        clf.fit(data, labels)
        self.parameters = clf.best_params_
        self.parameters["repeat"] = True
        # self.parameters["random_state"] = None
        return clf
        # clf = super(HMMGMMClassifier, self).tuneParameters(data, labels)
        # return clf


class TimeSeriesForest(TimeSeriesClassifier):
    def _getHyperparameters(self):
        return {
            'n_windows': [1, 3, 5, 7],
            "min_window_size": [1, 0.01, 0.03, 0.05],
        }

    def _createModel(self, **kwargs):
        return pyts.classification.TimeSeriesForest(random_state=RANDOM_STATE, **kwargs)


class TimeSeriesBagOfFeatures(TimeSeriesClassifier):
    def _getHyperparameters(self):
        return {
            'n_estimators': [250, 500, 750, 1000],
            'min_subsequence_size': [0.25, 0.5, 0.75],
            'min_interval_size': [0.1, 0.2, 0.3],
            'bins': [5, 10, 15],
        }

    def _createModel(self, **kwargs):
        return pyts.classification.TSBF(random_state=RANDOM_STATE, **kwargs)


class BagOfSFASymbolsVecSpace(TimeSeriesClassifier):
    def _getHyperparameters(self):
        return {
            'window_size': [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
            'word_size': [4, 8, 12],
            'n_bins': [4, 12, 20],
            'strategy': ['uniform', 'quantile', 'normal', 'entropy'],
        }

    def _createModel(self, **kwargs):
        return pyts.classification.BOSSVS(**kwargs)


class SymbolicAggreApproxVecSpace(TimeSeriesClassifier):
    def _getHyperparameters(self):
        return {
            'window_size': [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
            'word_size': [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
        }

    def _createModel(self, **kwargs):
        return pyts.classification.SAXVSM(**kwargs)
