import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def getXY(filename):

    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    data = np.asarray(data)
    # keras uses only the last part of data as validation so shuffle to get
    # every type in validation set
    shuffled = np.random.permutation(np.transpose(data))
    shuffled = np.transpose(shuffled)
    data = shuffled.tolist()


    x = [np.asarray(data[0]), np.asarray(data[1]), np.asarray(data[2])]
    y_data = data[3]

    # turn y_data to one-hot encoding
    encoder = LabelEncoder()
    encoder.fit(y_data)
    y_encoded = encoder.transform(y_data)
    y = np_utils.to_categorical(y_encoded)

    categories = len(encoder.classes_)

    return x, y, categories