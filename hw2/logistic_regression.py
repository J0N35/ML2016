# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import time
from sys import argv

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def importData(path, ratio = 0.99):
    data = pd.read_csv(path,header=None).set_index(0) # import dataset
    # parse dataset into training / testing dataset with ratio
    train = data.sample(frac=ratio)
    test = data.drop(train.index)
    return train, test

def parseData(data):
    data_amount, column_amount = data.shape
    feature_amount = column_amount - 1
    label = data.ix[:,column_amount].copy() # extract last column as label
    feature = data.ix[:,:feature_amount].copy() # extract other columns as features
    return feature, label

def crossEntropy(x, y):
    s = -(np.dot(y, np.log(x))+np.dot((1-y), np.log(1-x)))
    return s

def rms(x):
    return np.sqrt(x**2 + 1.0)

def featureScaling(feature):
    fmean = np.mean(feature)
    fsd = np.sqrt(np.mean(np.square(feature)) - np.square(fmean))
    fss = (feature - fmean) / fsd
    return fmean, fsd, fss

def gradientDescent(x, y, iterations, alpha, theta, bias, tx, ty):
    decay = pd.DataFrame({'Iterations' : [],'Loss' : []})
    s = 0.0
    for i in range(iterations):
        z = np.dot(x, theta) + bias
        # print("z: {}".format(z))
        p = sigmoid(z)
        # print("p: {}".format(p))
        if i%(0.1*iterations) == 0:
            loss = crossEntropy(p, y)
            s = crossValid(tx, ty, theta, bias)
            loss_entry = pd.DataFrame([[i, s]], columns=['Iterations', 'Accuracy'])
            decay = decay.append(loss_entry, ignore_index=True)
            print("Loss in iterations {}| CrossAccuracy:{}".format(i, s))
        theta_gradient = np.dot((p-y),x)
        bias_gradient = np.mean(np.square(p) - 1)
        theta = theta - np.multiply(alpha, theta_gradient) / rms(theta_gradient)
        bias = bias - np.multiply(alpha, bias_gradient) / (rms(bias_gradient)+1)
    return theta, bias, decay

def crossValid(feature, label, theta, bias):
    p = np.rint(sigmoid(np.dot(feature, theta) + bias))
    result = np.vstack((p, label)).T
    return accuracy(result)

def plot(data):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    data.plot(x='Iterations', y='Accuracy')

def startLogisticRegression(training_data, testing_data, iterations, alpha):
    data_amount, column_amount = training_data.shape
    feature_amount = column_amount - 1
    training_feature, training_label = parseData(training_data)
    testing_feature, testing_label = parseData(testing_data)
    # feature scaling
    training_mean, training_sd, training_feature = featureScaling(training_feature)
    testing_feature = (testing_feature - training_mean) / training_sd
    # initial theta, bias
    theta = (np.random.rand(feature_amount) - 0.5) * 0.01
    bias = 0.0
    # begin gradientDescent
    theta, bias, decay = gradientDescent(training_feature, training_label, iterations, alpha, theta, bias, testing_feature, testing_label)
    return theta, bias, decay, training_mean, training_sd

def exportModel(theta, bias, mean, sd, path):
    with open(path, "wb+") as model:
        pickle.dump((theta, bias, mean, sd), model)

def accuracy(result):
    c = 0
    for i in result:
        c = (c+1 if i[0]==i[1] else c+0)
    return c/result.shape[0]

if __name__ == "__main__":
    if (len(argv) < 2):
        filepath = "spam_train.csv"
        modelpath = 'model.pkl'
    else:
        filepath, modelpath = argv[1], argv[2]
    learning_rate, iterations= 5e-4, int(5e4)
    startTime = time.time() #   Timer start
    training_data, testing_data = importData(filepath)
    theta, bias, decay, mean, sd= startLogisticRegression(training_data, testing_data, iterations, learning_rate)
    print("Time used training: {}".format(time.time() - startTime))
    exportModel(theta, bias, mean, sd, modelpath) # Export model