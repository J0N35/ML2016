#!/usr/bin/python3
# coding: utf-8

import os

LABELED_DATA_PATH = "all_label.p"
UNLABELED_DATA_PATH = "all_unlabel.p"
TESTING_DATA_PATH = "test.p"

def import_data(path):
    import pickle
    
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def random_split(data, test_ratio):
    from sklearn.model_selection import ShuffleSplit
    
    rs = ShuffleSplit(n_splits=1, random_state=0, test_size=test_ratio, train_size=None)
    for train_id, test_id in rs.split(data):
        training_data_idx = train_id
        testing_data_idx = test_id
    return data[training_data_idx], data[testing_data_idx]

def random_shuffle(vector_a, vector_b):
    from numpy.random import permutation
    
    assert len(vector_a) == len(vector_b)
    idx = permutation(len(vector_a))
    return vector_a[idx], vector_b[idx]

def load_data(case=1, test_ratio=0.1, directory_path=os.getcwd()):
    import numpy as np
    
    if not os.path.isdir(directory_path):
        print("Invalid path")
        return
        
    # ===== import data =====
    if case == 0: # if import data for prediction
        testing_file = import_data(os.path.join(directory_path, TESTING_DATA_PATH)) # 10000 ID, 10000 labels, 10000 data * 3072 dim
        testing_data = np.roll(np.array(testing_file['data']).reshape(10000, 3, 32, 32), 2, axis=1)
        return testing_data
    
    label_data = np.roll(np.array(import_data(os.path.join(directory_path, LABELED_DATA_PATH))).reshape((10, 500, 3, 32, 32)), 2, axis=2) # 10 class * 500 pic per label * 3072 dim
    
    if case >= 1:
        unlabel_data = np.roll(np.array(import_data(os.path.join(directory_path, UNLABELED_DATA_PATH))).reshape((45000, 3, 32, 32)), 2, axis=1) # 45000 pic * 3072 dim
        testing_file = import_data(os.path.join(directory_path, TESTING_DATA_PATH)) # 10000 ID, 10000 labels, 10000 data * 3072 dim
        testing_data = np.roll(np.array(testing_file['data']).reshape(10000, 3, 32, 32), 2, axis=1)
    if case == 2:
        testing_data = load_data(0)
    # ===== preprocess data =====
    training_x, testing_x = np.zeros((1, 3, 32, 32)), np.zeros((1, 3, 32, 32)) # initial vectors
    training_y, testing_y = [], []
    for idx, i in enumerate(label_data): # for each labeled class (ensure dataset is balance)
        train_data, test_data = random_split(i, test_ratio) # random split n for testing data, other as training data
        training_x = np.vstack((training_x, train_data)) # stack training data for each class into one dataset
        testing_x = np.vstack((testing_x, test_data)) # stack testing data for each class into one dataset
        training_y = np.append(training_y, np.array([idx]*len(train_data))) # stack training label for each class into one vector
        testing_y = np.append(testing_y, np.array([idx]*len(test_data))) # stack testing label for each class into one vector
    # -----After random split-----
    training_x = np.delete(training_x, 0, 0) # remove the empty vector
    testing_x = np.delete(testing_x, 0, 0) # remove the empty vector
    training_y = training_y.reshape(-1, 1).astype(int) # transpose
    testing_y = testing_y.reshape(-1, 1).astype(int) # transpose
    # -----Shuffle data-----
    training_x, training_y = random_shuffle(training_x, training_y)
    testing_x, testing_y = random_shuffle(testing_x, testing_y)
    
    return training_x, training_y, testing_x, testing_y, unlabel_data