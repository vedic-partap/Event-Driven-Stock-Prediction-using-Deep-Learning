import random
import numpy as np
import operator
import json
import pandas as pd
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import model_from_json

def value2int_simple(y):
    label = np.copy(y)
    label[y < 0] = 0
    label[y >= 0] = 1
    return label


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



json_file = open('./model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./model.h5")
print("Loaded model from disk")

num_test_samples = 25000
batch_size = 512
test_filenames = "/media/walragatver/LENOVO/LinuxGit/stockFeatures/test.csv"
#Path where your test set is present

my_test_batch_generator = My_Generator(test_filenames,batch_size)

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


conf = loaded_model.evaluate_generator(my_test_batch_generator, steps=(int(np.ceil(num_test_samples/ (batch_size)))),
                         use_multiprocessing=False, verbose=1)
print(conf)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], conf[1]*100))
