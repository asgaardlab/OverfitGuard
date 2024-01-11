import json
import pickle


def loadJson(filepath):
    with open(filepath, "r") as fp:
        return json.load(fp)


def saveJson(data, filepath):
    with open(filepath, "w") as fp:
        return json.dump(data, fp)


def readPkl(dataPath):
    with open(dataPath, 'rb') as handle:
        return pickle.load(handle)


def savePkl(data, dataPath):
    with open(dataPath, 'wb') as handle:
        return pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def printPercentage(value, total, prompt=""):
    print(f"{value} / {total} = {value / total * 100}% {prompt}")

