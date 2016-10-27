# coding: utf-8

import numpy as np
import pandas as pd
import pickle
from logistic_regression import sigmoid

def loadModel(path):
	with open(path, "rb") as model:
		return pickle.load(model)

def exportPrediction(theta, bias, mean, sd, filepath, output_path):
    df = pd.read_csv(filepath,header=None).set_index(0)
    df = (df - mean) / sd 	#	feature standardization
    z = np.dot(df, theta) + bias
    p = sigmoid(z)
    result = pd.DataFrame(np.rint(p), columns=["label"], dtype=int)
    result.index.name = "id"
    result.index = result.index + 1
    result.to_csv(path_or_buf = output_path)
    print('Prediction exported to path:', output_path)

if __name__ == "__main__":
	from sys import argv
	if (len(argv) < 2):
		modelPath = "model.pkl"
		dataPath = "spam_test.csv"
		outputPath = "prediction.csv"
	else:
		modelPath, dataPath, outputPath = argv[1], argv[2], argv[3]

	theta, bias, mean, sd = loadModel(modelPath)
	exportPrediction(theta, bias, mean, sd, dataPath, outputPath)
