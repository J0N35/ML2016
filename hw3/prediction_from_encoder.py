#!/usr/bin/python3
# coding: utf-8

import numpy as np

def knn_train(training_x, training_y, model_path, directory_path, output_path):
	from keras.models import Model, load_model
	from sklearn.neighbors import KNeighborsRegressor

	import data_import

	encoder_model = load_model(model_path)
	training_data_encoded = encoder_model.predict(training_x) # encode training data
	training_data_encoded = training_data_encoded.reshape(len(training_data_encoded), -1) # reshape

	# --- knn training ---
	knn = KNeighborsRegressorKNeighborsRegressor(n_neighbors=10, leaf_size=2500)
	knn.fit(training_data_encoded, training_y)

	testing_x = data_import.load_data(case=0, test_ratio=0, directory_path=directory_path)
	# --- prediction ---
	prediction = knn.predict(testing_x).astype(int)
	prediction_id = np.array(range(len(prediction)))
	
	# --- combine and output ---
	result = np.vstack((prediction_id, prediction)).T
	np.savetxt(output_path, result, fmt='%i', delimiter=',', header='ID,class', comments='')

if __name__ == "__main__":
	from sys import argv
    from os import getcwd
    
    import data_import
    
    # directory_path = getcwd()
    # model_path = "cifar10_ae_model.hdf5"
    
    if (len(argv) < 2):
        directory_path = getcwd()
        model_path = "cifar10_encoder_model.hdf5"
        output_path = "ae_prediction.csv"
    else:
        directory_path, model_path = argv[1], argv[2], argv[3]
        
    # load data
    print("Loading Train Data...", end='')
    training_x, training_y, testing_x, testing_y, unlabel_data = data_import.load_data(case=1, test_ratio=0, directory_path=directory_path)
    print("Completed")
    
    knn_train(training_x, training_y, model_path, directory_path, output_path)
    print("=====Train Completed=====")