import random
import numpy as np
import operator
import json
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.utils import to_categorical
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import GRU
from keras import optimizers


import matplotlib.pyplot as plt
def value2int_simple(y):
    label = np.copy(y)
    label[y < 0] = 0
    label[y >= 0] = 1
    return label

# Loading 3 miniBatches Together
def My_Generator(fileName,batch_size):
    chunksize = batch_size
    while True:
        for chunk in pd.read_csv(fileName, chunksize=chunksize,sep=" "):
            batchFeatures = np.array(chunk.ix[:,:-1])
            batchFeatures = np.reshape(batchFeatures,(batchFeatures.shape[0],30,100))
            batchLabels = np.matrix(chunk.ix[:,-1]).T
            batchLabels = to_categorical(value2int_simple(batchLabels),num_classes=2).astype("int")
            batchLabels= np.matrix(batchLabels)
            yield batchFeatures,batchLabels


def BiLSTM():
    model = Sequential()
    model.add(Bidirectional(GRU(128, return_sequences=True) ,input_shape=(30, 100),merge_mode ='ave'))
    model.add(Bidirectional(GRU(64, return_sequences=True,activation='relu'),merge_mode ='ave'))
    model.add(BatchNormalization(axis=-1))
    #model.add(LSTM(128,return_sequences=True,activation='relu'))
    #model.add(BatchNormalization(axis=-1))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.2))
    model.add(Dense(16,activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dense(2, activation='softmax'))
    adam = optimizers.adam(lr=0.0003)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    return model

def main():
    model = BiLSTM()
    print(model.summary())
    # Parameters
    batch_size = 512
    training_filenames = "../stockFeatures/train.csv"
    validation_filenames = "../stockFeatures/validation.csv"
    test_filenames = "../stockFeatures/test.csv"
    num_training_samples = 100000
    num_validation_samples = 25000
    num_test_samples = 25000

    # Training and Validation Set Loader
    my_training_batch_generator = My_Generator(training_filenames, batch_size)
    my_validation_batch_generator =My_Generator(validation_filenames, batch_size)
    my_test_batch_generator = My_Generator(test_filenames,batch_size)
    my_test_batch_generator_one = My_Generator(test_filenames,batch_size)

    print(my_training_batch_generator)

    modelHistory = model.fit_generator(generator=my_training_batch_generator,
                      steps_per_epoch=(int(np.ceil(num_training_samples/ (batch_size)))),
                      epochs=100,
                      verbose=1,
                      validation_data = my_validation_batch_generator,
                      validation_steps = (int(np.ceil(num_validation_samples // (batch_size)))),
                      max_queue_size=32)

    conf = model.evaluate_generator(my_test_batch_generator, steps=(int(np.ceil(num_test_samples/ (batch_size)))),
                             use_multiprocessing=False, verbose=1)
    print(conf)
    print("%s: %.2f%%" % (model.metrics_names[1], conf[1]*100))
    conf = model.predict_generator(my_test_batch_generator_one, steps=(int(np.ceil(num_test_samples/ (batch_size)))), use_multiprocessing=False, verbose=1)
    print(conf)

    model_json = model.to_json()
    with open("./input/model.json", "w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5
    model.save_weights("./input/model.h5")
    print("Saved model to disk")
    print(modelHistory.history.keys())
    plt.plot(modelHistory.history['loss'])
    plt.plot(modelHistory.history['val_loss'])
    plt.title('model_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()
