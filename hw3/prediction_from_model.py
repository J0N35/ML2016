#!/usr/bin/python3
# coding: utf-8

def load_and_predict(directory_path, model_path, output_path):
    import numpy as np
    from keras.models import Sequential, load_model
    
    import data_import
    
    testing_data = data_import.load_data(case=0, directory_path=directory_path) # load data
    model = load_model(model_path) # load model
    prediction = model.predict_classes(testing_data) # make prediction
    prediction_id = np.array(range(len(prediction)))
    result = np.vstack((prediction_id, prediction)).T
    np.savetxt(output_path, result, fmt='%i', delimiter=',', header='ID,class', comments='')
    
if __name__ == "__main__":
    from sys import argv
    from os import getcwd
    
    if (len(argv) < 2):
        directory_path = getcwd()
        model_path = "cifar10_cnn_keras_model.hdf5"
        output_path = "prediction.csv"
    else:
        directory_path, model_path, output_path = argv[1], argv[2], argv[3]
        
    load_and_predict(directory_path, model_path, output_path)
    print("---Prediction Completed---")