#!/usr/bin/python3
# coding: utf-8

import numpy as np

def autoencode(training_X, nb_epoch=500, filepath='cifar10_autoencoder_model.hdf5'):
    from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
    from keras.optimizers import Adam
    from keras.models import Model
    from keras.layers import PReLU
    from keras.regularizers import l1
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    
    batch_size = int(len(training_X)/10)
    input_img = Input(shape=(3, 32, 32))

    x = Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2), W_regularizer=l1(0.01))(input_img)
    x = PReLU(init='zero', weights=None)(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    
    x = Convolution2D(16, 3, 3, border_mode='same')(x)
    x = PReLU()(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)
    
#     x = Convolution2D(16, 3, 3, border_mode='same')(x)
#     x = PReLU()(x)
#     encoded = MaxPooling2D((2, 2), border_mode='same')(x)

    x = UpSampling2D((2, 2))(encoded)
    x = Convolution2D(16, 3, 3, border_mode='same')(x)
    x = PReLU()(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = PReLU()(x)
    x = UpSampling2D((2, 2))(x)
    
#     x = Convolution2D(64, 3, 3, border_mode='same')(x)
#     x = PReLU()(x)
#     x = UpSampling2D((2, 2))(x)
    
    decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)
    
    training_X = training_X.astype('float32') / 255.
    
    autoencoder = Model(input_img, decoded)
    encoder_model = Model(input_img, encoded)
    autoencoder.compile(loss='mse', optimizer='Adam', metrics=['mean_squared_error'])
    
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=int(nb_epoch*0.1), verbose=1)
#     print(autoencoder.summary())
    autoencoder.fit(training_X, training_X, batch_size=batch_size, nb_epoch=nb_epoch, 
                    verbose=2, callbacks=[checkpointer, earlystopping], validation_split=0.05, validation_data=None, 
                    shuffle=True, class_weight=None, sample_weight=None)
    
    return encoder_model

def initial_training(training_x, unlabel_data, model_path="cifar10_encoder_model.hdf5"):
    from keras.models import Model, load_model
    from sklearn.neighbors import KNeighborsRegressor

    training_data = np.vstack((training_x, unlabel_data)) # combine (un)labeled data into larger dataset
    encoder_model = autoencode(training_data) # train encoder
    encoder_model.save(model_path) # save encoder model

#     training_data_encoded = encoder_model.predict(training_x) # encode labeled data
#     training_data_encoded = training_data_encoded.reshape(len(training_data_encoded), -1)
#     training_data_label = create_label()

#     knn = KNeighborsRegressorKNeighborsRegressor(n_neighbors=5, leaf_size=1000)
#     knn.fit(training_data_encoded, training_data_label)

# def create_label():
#     label = []
#     for i in range(10):
#         for j in range(500):
#             label.append(i)
#     return np.array(label)

if __name__ == "__main__":
    from sys import argv
    from os import getcwd
    
    import data_import
    
    # directory_path = getcwd()
    # model_path = "cifar10_ae_model.hdf5"
    
    if (len(argv) < 2):
        directory_path = getcwd()
        model_path = "cifar10_encoder_model.hdf5"
    else:
        directory_path, model_path = argv[1], argv[2]
        
    # load data
    print("Loading Data...", end='')
    training_x, training_y, testing_x, testing_y, unlabel_data = data_import.load_data(case=1, test_ratio=0, directory_path=directory_path)
    print("Completed")
    
    initial_training(training_x, unlabel_data, model_path)
    print("=====Train Completed=====")

