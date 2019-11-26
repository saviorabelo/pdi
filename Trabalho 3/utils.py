import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer


class Data:
    def numbers():
        dataset = pd.read_csv('Datasets/ocr_car_numbers_rotulado.txt', sep=' ', header=None)
        Y = dataset.iloc[:,-1:] # get last column
        X = dataset.iloc[:,:-1] # remove last column

        # Transform classes
        lb = LabelBinarizer()
        lb.fit(np.unique(Y))
        Y = lb.transform(np.array(Y))

        return np.array(X), Y, lb
