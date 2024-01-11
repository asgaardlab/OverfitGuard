import numpy as np


def standarize(data):
    data = np.array(data)
    assert data.ndim == 1
    # print(data)
    scale = data.std()
    if scale == 0:
        scale = 1.0
    return ((data - data.mean()) / scale).tolist()


def stretchData(data, dstLen):
    size = len(data)
    xloc = np.arange(size)
    new_xloc = np.linspace(0, size, dstLen)
    stretched_data = np.interp(new_xloc, xloc, data)
    return stretched_data
