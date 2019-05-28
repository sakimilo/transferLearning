import os
import shutil

import tensorflow as tf
from tensorflow import keras

from logs import logDecorator as lD 
import jsonref
import numpy as np
import pickle
from tqdm import tqdm

import PIL
import matplotlib.pyplot as plt

config      = jsonref.load(open('../config/config.json'))
logBase     = config['logging']['logBase'] + '.modules.data.getData'
dataFolder  = '../data/raw_data/'

@lD.log(logBase + '.getInputDataDict')
def getInputDataDict(logger, resize_shape=(56, 56, 3)):

    try:

        shape2D       = resize_shape[:2]
        channelSize   = resize_shape[-1]

        dataDict      = mnistFashion()
        inputDataDict = {}

        for dataName in ['train_images', 'test_images']:
            tmpArr      = []
            imageStack  = dataDict[ dataName ]
            for img in tqdm(imageStack):
                img         = img / 255.0
                img_resized = PIL.Image.fromarray(img).resize(size=shape2D)
                img_resized = np.array(img_resized)
                tmpArr.append( img_resized )
            tmpArr      = np.stack( tmpArr )
            tmpArr      = np.stack([ tmpArr ] * channelSize, axis=-1)
            inputDataDict[dataName] = tmpArr

        return inputDataDict

    except Exception as e:

        logger.error('Unable to generate train data \n{}'.format(str(e)))

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

        visualiseArray( train_images[imageIndex] )

    except Exception as e:

        logger.error('Unable to show one image \n{}'.format(str(e)))

@lD.log(logBase + '.visualiseArray')
def visualiseArray(logger, img):

    try:

        plt.figure()
        plt.imshow( img )
        plt.colorbar()
        plt.grid(False)
        plt.show()

    except Exception as e:

        logger.error('Unable to visualise image array \n{}'.format(str(e)))

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

        stackedImg    = np.stack([ train_images[i] for i in range(N) ], axis=-1)
        labels        = [ class_names[train_labels[i]] for i in range(N) ]

        visualiseStackedArray( stackedImg, labels )

    except Exception as e:

        logger.error('Unable to show one image \n{}'.format(str(e)))

@lD.log(logBase + '.visualiseStackedArray')
def visualiseStackedArray(logger, stackedImg, xlabels=None, cmap=plt.cm.binary):

    try:

        N         = stackedImg.shape[-1]
        sqrtN     = np.ceil(np.sqrt(N))

        if sqrtN > 10:
            rowN, colN = np.ceil( N / 10 ), 10
        else:
            rowN, colN = sqrtN, sqrtN

        plt.figure(figsize=(10, rowN))
        for i in range(N):
            plt.subplot(rowN, colN, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow( stackedImg[:, :, i], cmap=cmap )

            if xlabels is not None:
                plt.xlabel( xlabels[i] )
        
        plt.tight_layout()
        plt.show()

    except Exception as e:

        logger.error('Unable to visualise stacked array \n{}'.format(str(e)))

@lD.log(logBase + '.main')
def main(logger, resultsDict):

    try:

        print('getting numpy MNIST data dictionary')
        dataDict      = mnistFashion()
        print('keys', dataDict.keys())

        print('getting stacked & resized MNIST data array with channels')
        inputDataDict = getInputDataDict( resize_shape=(56, 56, 3) )
        print( inputDataDict['train_images'].shape, inputDataDict['train_images'].max(), inputDataDict['train_images'].min() )
        print( inputDataDict['test_images'].shape, inputDataDict['test_images'].max(), inputDataDict['test_images'].min() )

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