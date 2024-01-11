#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np


# Reference: https://github.com/keras-team/keras/blob/v2.9.0/keras/callbacks.py#L1745-L1893
class EarlyStopping(object):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(EarlyStopping, self).__init__()

        self.stop_training = False
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = -1
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        self.on_train_begin()

    def on_train_begin(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = -1
        self.stop_training = False
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def check_step(self, epoch, current):
        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.stop_training = True

    def get_stop_epoch(self):
        return self.stopped_epoch

    def get_best_epoch_loss(self):
        return self.best

    def get_best_epoch(self):
        return self.best_epoch

    def is_stop(self):
        return self.stop_training

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)


def detectEarlyStop(epochs, valErrors, patience=5):
    es = EarlyStopping(patience=patience)
    # best_epoch = -1
    for epoch, val in zip(epochs, valErrors):
        es.check_step(epoch, val)
        if es.is_stop():
            break
    else:
        es.stopped_epoch = epoch
    return es.is_stop(), es.get_stop_epoch(), es.get_best_epoch(), es.best
    #     if es.is_stop() and es.get_best_epoch() != epochs[-1]:
    #         best_epoch = es.get_best_epoch()
    #         break
    # return best_epoch


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import json
    test_errors = np.loadtxt("../../../out/tensorflow_lr_only/lenet5/py/testing_errors_0.txt")
    # with open("../../../out/tensorflow_lr_only/lenet5/py/testing_errors_0.txt", "r") as f:
    #     test_errors = json.loads(f)
    stop_epoch = detectEarlyStop(list(range(len(test_errors))), test_errors)
    print(stop_epoch)
    plt.plot(test_errors)
    plt.show()
