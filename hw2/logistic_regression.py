# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import time

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def importData(path, ratio = 0.95):
    data = pd.read_csv(path,header=None).set_index(0) # import dataset
    # parse dataset into training / testing dataset with ratio
    train = data.sample(frac=ratio)
    test = data.drop(train.index)
    return train, test

def crossEntropy(x, y):
    s = -(np.dot(y, np.log(x))+np.dot((1-y), np.log(1-x)))
    return s

def rms(x):
    return np.sqrt(x**2 + 1.0)

def crossValid(feature, label, theta, bias):
    p = sigmoid(np.dot(feature, theta) + bias)
    p = np.rint(p)
    result = np.vstack((p, label)).T
    return accuracy(result)

def accuracy(result):
    c = 0
    for i in result:
        c = (c+1 if i[0] == i[1] else c+0)
    return c/result.shape[0]

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
        p = sigmoid(z)
        if i%(0.1*iterations) == 0:
            s = crossValid(tx, ty, theta, bias)
            loss_entry = pd.DataFrame([[i, s]], columns=['Iterations', 'Loss'])
            decay = decay.append(loss_entry, ignore_index=True)
        theta_gradient = np.dot((p-y),x)
        bias_gradient = np.mean(np.square(p) - 1)
        theta = theta - np.multiply(alpha, theta_gradient) / rms(theta_gradient)
        bias = bias - np.multiply(alpha, bias_gradient) / (rms(bias_gradient)+1)
    return [theta, bias, decay]

def exportTestFile(theta, bias, mean, sd, filepath = "spam_test.csv"):
    df = pd.read_csv(filepath,header=None).set_index(0)
    df = (df - mean) / sd
    z = np.dot(df, theta) + bias
    p = sigmoid(z)
    result = pd.DataFrame(np.rint(p), columns=["label"], dtype=int)
    result.index.name = "id"
    result.index = result.index + 1
    result.to_csv(path_or_buf = "output_path.csv")
    print('Test Result exported to path:', filepath)

def plot(data):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    data.plot(x='Iterations', y='Loss')

def parseData(data):
    data_amount, column_amount = data.shape
    feature_amount = column_amount - 1
    label = data.ix[:,column_amount].copy() # extract last column as label
    feature = data.ix[:,:feature_amount].copy() # extract other columns as features
    return feature, label

def startLogisticRegression(iterations, alpha, filepath):
    training_data, testing_data = importData(filepath)
    data_amount, column_amount = training_data.shape
    feature_amount = column_amount - 1
    training_feature, training_label = parseData(training_data)
    testing_feature, testing_label = parseData(testing_data)
    # feature scaling
    training_mean, training_sd, training_feature = featureScaling(training_feature)
    testing_feature = (testing_feature - training_mean) / training_sd
    # initial theta, bias
    loadmodel = False
    if loadmodel:
        try:
            with open("model.pkl", "rb") as model:
                theta, bias = pickle.load(model)
                print(theta, bias)
        except:
            print("Cannot load model")
            theta = (np.random.rand(feature_amount) - 0.5) * 0.02
            bias = 0.0
            print("para reset")
    else:
        theta = (np.random.rand(feature_amount) - 0.5) * 0.02
        bias = 0.0
    # begin gradientDescent
    theta, bias, decay = gradientDescent(training_feature, training_label, iterations, alpha, theta, bias, testing_feature, testing_label)
    return [theta, bias, decay, training_mean, training_sd]

def exportModel(model, path):
    with open(path, "wb+") as file:
        pickle.dump(model, file)
    print("Exported Models to path: {}".format(path))

if __name__ == "__main__":
    from sys import argv
    if (len(argv) < 2):
        filepath = "spam_train.csv"
        modelpath = "models.p"
    else:
        filepath, modelpath = argv[1], argv[2]
    learning_rate, iterations = 1e-3, int(2e4)

    startTime = time.time()
    acc = []
    model = []
    for i in range(10):
        print("Processing Model {}...".format(i+1))
        theta, bias, decay, mean, sd= startLogisticRegression(iterations, learning_rate, filepath)
        model.append((theta, bias, decay, mean, sd))
        acc.append(decay['Loss'].iloc[-1])
    exportModel(model, modelpath)
    print("Time used training: {}".format(time.time() - startTime))
    print("Average Accuracy:{}".format(np.mean(weight)))
