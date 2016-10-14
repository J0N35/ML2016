# coding: utf-8

import random
import pandas as pd
import numpy as np

def importData(path):
    from os.path import abspath
    path = abspath('train.csv') # create abs path
    df = pd.read_csv(path, encoding='BIG5') # import data
    # tranform data
    df = df.drop('測站', axis = 1) # delete unused column
    df = df.rename(columns={'日期': 'date', '測項': 'para'}) # change column name into english, easier for modified
    # extract date for grouping
    ndf = df.set_index('date')
    dateList = pd.Series(ndf.index).drop_duplicates()
    df = df.set_index(['date','para'])
    # Change dataframe data type from str object to float/int
    df = df.T
    for i in dateList:
        df[i] = df[i].replace({'RAINFALL':'NR'}, 0.0)
    df = df.apply(pd.to_numeric)
    # extract data per day and concatenate together
    x_list, y_list = [], []
    windowSize = 10
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
    data = [np.array(x_list), np.array(y_list)]
    return data

def linearRegression(feature, label, iterations, learning_rate):
    loss_decay = pd.DataFrame({'Iterations' : [],'Loss' : []})
    data_amount, feature_amount = feature.shape
    bias, theta = 0.0, np.zeros(feature_amount)
    feature_transpose = feature.T
    for i in range(iterations):
        prediction = np.dot(feature, theta) + bias
        error = label - prediction
        loss = np.square(error)
        if i%(iterations*0.1) == 0:
            cost = np.mean(np.sqrt(loss))
            loss_entry = pd.DataFrame([[i, cost]], columns=['Iterations', 'Loss'])
            loss_decay = loss_decay.append(loss_entry, ignore_index=True)
            print("Iteration %d | Cost: %f" % (i, cost))
        bias_gradient = np.mean(-2 * error)
        theta_gradient = -2 * np.dot(error, feature)
        # update bias and theta
        bias = bias - learning_rate*bias_gradient
        theta = theta - learning_rate*theta_gradient
    return [bias, theta, loss_decay]

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
    test_df = test_df.set_index(['id','para']).T

    for i in idList:
            test_df[i] = test_df[i].replace({'RAINFALL':'NR'}, 0.0)

    test_df = test_df.apply(pd.to_numeric)

    pm_25_id = [i for i in idList]
    pm_25_value = list()
    for i in idList:
        test_x = np.array(test_df[i].values)
        test_y = np.dot(test_x.flatten(), theta) + bias
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
    output_path = 'linear_regression.csv'
    i, n = int(1e5), 1e-10
        
    x, y = importData(training_data)
    bias_result, theta_result , loss_decay = linearRegression(x, y, i, n) # training
    plot(loss_decay) # plot the loss over iterations (if required)
    testData(theta_result, bias_result, testing_data, output_path) # generate test prediction
