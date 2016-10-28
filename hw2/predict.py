import numpy as np
import pandas as pd
from time import time
import pickle
from logistic_regression import sigmoid

def loadModel(path):
    with open(path, "rb") as file:
        return pickle.load(file)

def ensembleTest(theta, bias, mean, sd, path):
    df = pd.read_csv(path,header=None).set_index(0)
    df = (df - mean) / sd
    z = np.dot(df, theta) + bias
    return sigmoid(z)

def exportPrediction(result, weight, path):
    result = np.average(result, axis = 0, weights = weight)
    result = pd.DataFrame(np.rint(result), columns=["label"], dtype=int)
    result.index.name = "id"
    result.index = result.index + 1
    result.to_csv(path_or_buf = path)

if __name__ == "__main__":
    from sys import argv
    if (len(argv) < 2):
        modelpath = "models.p"
        datapath = "spam_test.csv"
        outputpath = "prediction.csv"
    else:
        modelpath, datapath, outputpath = argv[1], argv[2], argv[3]

    startTime = time()
    models = loadModel(modelpath)
    accuracy = []
    result = []
    for i in models:
        theta, bias, decay, mean, sd = i   
        accuracy.append(decay['Loss'].iloc[-1])
        result.append(ensembleTest(theta, bias, mean, sd, datapath))
    exportPrediction(result, accuracy, outputpath)
    print('Test Result exported to path:', outputpath)
    print("Time used testing: {}".format(time() - startTime))