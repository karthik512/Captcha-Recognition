from PIL import Image
import numpy as np
import os
from src.Constants import *


class DataLoader(object):

    def __init__(self, directory):
        self.trainDirectory = directory + '\\TrainNew'
        self.validationDirectory = directory + '\\ValidationNew'
        self.testDirectory = directory + '\\TestNew'

    def load_current_data(self, directory, noOfDataToLoad):
        X_Img = []
        Y_Img = []
        currentDir = os.listdir(directory)
        print(' Loading from Directory :: ' + directory)
        i = 0
        if os.path.isfile(directory + 'XData.npy') and os.path.isfile(directory + 'YData.npy'):
            print('Loading Data from File')
            x_arr = np.load(directory + 'XData.npy')
            y_arr = np.load(directory + 'YData.npy')
            x_arr[:], y_arr[:] = randomize(x_arr, y_arr)
            x_arr, y_arr = x_arr[:noOfDataToLoad], y_arr[:noOfDataToLoad]
            return x_arr, y_arr

        for item in currentDir:
            #if i > 500:
            #    break
            if '.png' in item:
                image = Image.open(directory + '\\' + item)
                X = np.array(image).flatten()   #160 * 60 * 3
                Y = item.split('.')[0]
                TY = np.vstack([self.char_to_vec(ch) for ch in Y]).flatten()
                X_Img.append(X)
                Y_Img.append(TY)
                i = i + 1
        x_arr = np.array(X_Img)
        y_arr = np.array(Y_Img)

        x_arr[:], y_arr[:] = randomize(x_arr, y_arr)

        np.save(directory + 'XData.npy', x_arr)
        np.save(directory + 'YData.npy', y_arr)

        x_arr, y_arr = x_arr[:noOfDataToLoad], y_arr[:noOfDataToLoad]
        return x_arr, y_arr

    def load_training_data(self):
        return self.load_current_data(self.trainDirectory, TRAIN_DATA_TO_LOAD)

    def load_validation_data(self):
        return self.load_current_data(self.validationDirectory, VALIDATION_DARA_TO_LOAD)

    def load_test_data(self):
        return self.load_current_data(self.testDirectory, TEST_DATA_TO_LOAD)

    def load_data(self):
        training_data = self.load_training_data()
        validation_data = self.load_validation_data()
        test_data = self.load_test_data()
        return training_data, validation_data, test_data

    def char_to_vec(self, ch):
        y = np.zeros((CLASSES,))
        y[CHARS.index(ch)] = 1.0
        return y
