# coding: utf-8

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def importData(path):
    from os.path import abspath

    path = abspath('train.csv')
    df = pd.read_csv(path, encoding='BIG5') # import data
    # tranform data
    df = df.drop('測站', axis = 1)
    df = df.rename(columns={'日期': 'date', '測項': 'para'})
    # extract date for grouping
    ndf = df.set_index('date')
    dateList = pd.Series(ndf.index).drop_duplicates()
    df = df.set_index(['date','para'])
    # extract data per day and concatenate together
    windowSize = 10
    df = df.T
    for i,t in df:
        df[i] = df[i].replace({'RAINFALL':'NR'}, 0.0).apply(pd.to_numeric)

    x_list, y_list = [], []
    for i in dateList:
        ndf = df[i]
        for j in range(0, ndf.shape[0]+1-windowSize):
            temp = ndf.iloc[j:windowSize+j]
            window_x = []
            for k in temp.index[:-1]:
                window_x = np.append(window_x, temp.loc[k].tolist()) # features from 9 time slot
            window_y = temp.iloc[-1]['PM2.5']
            x_list.append(window_x)
            y_list.append(window_y)

    x_list, y_list = np.array(x_list), np.array(y_list)
    data = np.vstack((y_list, x_list.T)).T
    return data

def linearRegression(data, iterations, learning_rate):
    loss_decay = pd.DataFrame({'Iterations' : [], 'Loss' : []})
    data_amount, feature_amount = data.shape[0], data.shape[1]-1
    bias = 0.0
    m, n = np.zeros(feature_amount), np.zeros(feature_amount)

    for i in range(iterations):
        np.random.shuffle(data) # shuffle data
        label, feature = data[:,0], data[:,1:] # split data into label and feature sets

        prediction = np.dot(feature**2, m) + np.dot(feature, n) + bias
        error = label - prediction
        loss = np.square(error)
        if i%(iterations*0.1) == 0:
            cost = abs(np.mean(np.sqrt(loss)))
            loss_entry = pd.DataFrame([[i, cost]], columns=['Iterations', 'Loss'])
            loss_decay = loss_decay.append(loss_entry, ignore_index=True)
            print("Iteration %d | Cost: %f" % (i, cost))

        for k,j in enumerate(error):
            indi_m_gradient = -2 * np.dot(j, feature[k]**2) #-2x^2(y-predict)
            indi_n_gradient = 2 * np.dot(j, feature[k]) # 2x(y-predict)
            indi_bias_gradient = np.sum(-2 * j) # 2(y-predict)
            # update theta and bias
            bias = updateGradient(bias, learning_rate, indi_bias_gradient)
            m = updateGradient(m, learning_rate, indi_m_gradient)
            n = updateGradient(n, learning_rate, indi_n_gradient)
    theta = np.vstack((m, n)).T
    return [bias, theta, loss_decay]

def updateGradient(w, n, gradient):
    return w - n*gradient

def testData(theta, bias, testing_data, output_path):
    from os.path import abspath
    
    testingDataPath = abspath(testing_data)
    output_path = abspath(output_path)

    test_df = pd.read_csv(testingDataPath, encoding='BIG5')
    test_df = test_df.T
    test_df = test_df.reset_index().T
    test_df.columns = ['id','para', '0', '1', '2', '3', '4', '5', '6', '7', '8']
    test_df = test_df.reset_index(drop=True)
    # extract date for grouping
    ndf = test_df.set_index('id')
    idList = pd.Series(ndf.index).drop_duplicates()
    test_df = test_df.set_index(['id','para'])
    pm_25_id = [i for i in idList]
    pm_25_value = list()
    for i in idList:
        test_x = np.array(test_df.T[i].replace({'RAINFALL':'NR'}, 0.0).values.flatten(),dtype= float)
        test_y = np.dot(test_x**2, theta[:,0]) + np.dot(test_x, theta[:,1]) + bias
        pm_25_value.append(int(test_y))
    resultDF = pd.DataFrame(data=pm_25_value, index=pm_25_id, columns=['value'])
    resultDF.index.name = 'id'
    resultDF.to_csv(path_or_buf = output_path)
    print('Test Result exported to path:', output_path)

def plot(data):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    data.plot(x='Iterations', y='Loss')

if __name__ == '__main__':
    training_data = 'train.csv'
    testing_data = 'test_X.csv'
    output_path = 'kaggle_best.csv'
    i, n = int(1e4), 5e-15
    
    dataset = importData(training_data)    
    bias_result, theta_result, loss_decay = linearRegression(dataset, i, n) # training
    plot(loss_decay) # plot the loss over iterations (if required)
    testData(theta_result, bias_result, testing_data, output_path) # generate test prediction  
