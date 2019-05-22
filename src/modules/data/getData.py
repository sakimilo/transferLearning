import os
import shutil

import tensorflow as tf
from tensorflow import keras

from logs import logDecorator as lD 
import jsonref
import numpy as np
import pickle

import matplotlib.pyplot as plt

config      = jsonref.load(open('../config/config.json'))
logBase     = config['logging']['logBase'] + '.modules.data.getData'
dataFolder  = '../data/raw_data/'

@lD.log(logBase + '.mnistFashion')
def mnistFashion(logger):

    try:

        cacheFilePath = os.path.join(dataFolder, 'mnist_fashion.pkl')

        if not os.path.exists( cacheFilePath ):

            fashion_mnist = keras.datasets.fashion_mnist
            (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

            shutil.rmtree('/Users/lingyit/.keras/datasets/fashion-mnist/')

            dataDict      = {
                'train_images' : train_images,
                'train_labels' : train_labels,
                'test_images'  : test_images,
                'test_labels'  : test_labels
            }

            pickle.dump(dataDict, open(cacheFilePath, 'wb'))

        else:
            dataDict = pickle.load(open(cacheFilePath, 'rb'))

        return dataDict

    except Exception as e:

        logger.error('Unable to get mnist Fashion data \n{}'.format(str(e)))

@lD.log(logBase + '.showOneImg')
def showOneImg(logger, imageIndex):

    try:

        dataDict      = mnistFashion()
        train_images  = dataDict['train_images']
        train_labels  = dataDict['train_labels']
        test_images   = dataDict['test_images']
        test_labels   = dataDict['test_labels']

        plt.figure()
        plt.imshow(train_images[imageIndex])
        plt.colorbar()
        plt.grid(False)
        plt.show()

    except Exception as e:

        logger.error('Unable to show one image \n{}'.format(str(e)))

@lD.log(logBase + '.showMultipleImgs')
def showMultipleImgs(logger, N):

    try:

        dataDict      = mnistFashion()
        train_images  = dataDict['train_images']
        train_labels  = dataDict['train_labels']
        test_images   = dataDict['test_images']
        test_labels   = dataDict['test_labels']
        
        class_names   = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        sqrtN         = np.ceil(np.sqrt(N))

        plt.figure(figsize=(10,10))
        for i in range(N):
            plt.subplot(sqrtN, sqrtN, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[train_labels[i]])
        
        plt.tight_layout()
        plt.show()

    except Exception as e:

        logger.error('Unable to show one image \n{}'.format(str(e)))

@lD.log(logBase + '.main')
def main(logger, resultsDict):

    try:

        dataDict = mnistFashion()

        print(dataDict)

    except Exception as e:

        logger.error('Unable to run main \n{}'.format(str(e)))

if __name__ == '__main__':

    print('tf.__version__ :', tf.__version__)

    ### -------------------------------------------------
    ### Give it a try run on getting mnist fashion data
    ### -------------------------------------------------
    print('try fetching data..')
    dataDict  = mnistFashion()
    
    for dataName in dataDict:
        data  = dataDict[dataName]
        print(dataName, type(data), data.shape)