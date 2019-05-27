import os
import shutil
import tensorflow as tf
from tensorflow import keras

from logs import logDecorator as lD 
import jsonref
import numpy as np
import pickle
import warnings

from modules.data import getData

config      = jsonref.load(open('../config/config.json'))
logBase     = config['logging']['logBase'] + '.modules.model.getPretrained'

### turn off tensorflow info/warning/error or all python warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

@lD.log(logBase + '.model')
def modelImageNet(logger, modelName, weightsFile=None, input_shape=(224, 224, 3)):

    try:

        if weightsFile is not None:
            weights = weightsFile
        else:
            weights = 'imagenet'

        if   modelName == 'Xception':
            base_model  = keras.applications.xception.Xception(input_shape=input_shape, include_top=False, weights=weights)
        elif modelName == 'VGG16':
            base_model  = keras.applications.vgg16.VGG16(input_shape=input_shape, include_top=False, weights=weights)
        elif modelName == 'VGG16_includeTop':
            base_model  = keras.applications.vgg16.VGG16(input_shape=input_shape, include_top=True, weights=weights)
        elif modelName == 'VGG19':
            base_model  = keras.applications.vgg19.VGG19(input_shape=input_shape, include_top=False, weights=weights)
        elif modelName == 'ResNet50':
            base_model  = keras.applications.resnet50.ResNet50(input_shape=input_shape, include_top=False, weights=weights)
        elif modelName == 'InceptionV3':
            base_model  = keras.applications.inception_v3.InceptionV3(input_shape=input_shape, include_top=False, weights=weights)
        elif modelName == 'InceptionResNetV2':
            base_model  = keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=input_shape, include_top=False, weights=weights)
        elif modelName == 'MobileNet':
            base_model  = keras.applications.mobilenet.MobileNet(input_shape=input_shape, include_top=False, weights=weights)
        elif modelName == 'DenseNet':
            base_model  = keras.applications.densenet.DenseNet121(input_shape=input_shape, include_top=False, weights=weights)
        elif modelName == 'NASNet':
            base_model  = keras.applications.nasnet.NASNetMobile(input_shape=input_shape, include_top=False, weights=weights)

        return base_model

    except Exception as e:

        logger.error('Unable to get model: {} \n{}'.format(modelName, str(e)))

@lD.log(logBase + '.outputTensorBoard')
def outputTensorBoard(logger, subfolder=None):

    try:

        tfboardFolder = '../notebooks/tensorlog/'

        if subfolder is not None:
            tfboardFolder = os.path.join(tfboardFolder, subfolder)

        if os.path.exists(tfboardFolder):
            shutil.rmtree(tfboardFolder)

        os.makedirs(tfboardFolder)

        with tf.Session() as sess:
            tfWriter = tf.summary.FileWriter(tfboardFolder, sess.graph)
            tfWriter.close()

    except Exception as e:

        logger.error('Unable to output tensorboard \n{}'.format(str(e)))

@lD.log(logBase + '.visualise_graph')
def visualise_graph(logger, modelName, subfolder=None):

    try:
        tf.keras.backend.clear_session()

        tfboardFolder = '../notebooks/tensorlog/'

        if subfolder is not None:
            tfboardFolder = os.path.join(tfboardFolder, subfolder)

        if os.path.exists(tfboardFolder):
            shutil.rmtree(tfboardFolder)

        os.makedirs(tfboardFolder)

        img           = np.random.randint(0, 5, (1, 224, 224, 3))

        modelDict     = getModelFileDict()
        modelLoaded   = modelImageNet(modelName, modelDict[modelName])

        with tf.Session() as sess:
            tfWriter   = tf.summary.FileWriter(tfboardFolder, sess.graph)
            tfWriter.close()

    except Exception as e:

        logger.error('Unable to write graph into tensorboard\n{}'.format(str(e)))

@lD.log(logBase + '.visualise_layers')
def visualise_layers(logger, model, listOfTensorNodes, inputData):

    try:


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            outputResults = sess.run( listOfTensorNodes,
                                      feed_dict={
                                                    'input_1:0' : inputData
                                                })
        for res, tf_node in zip(outputResults, listOfTensorNodes):
            print('-'*50)
            print('node: {}; shape: {}'.format(tf_node, res[0].shape))
            getData.visualiseStackedArray(res[0], cmap=None)

    except Exception as e:

        logger.error('Unable to visualise layers \n{}'.format(str(e)))

@lD.log(logBase + '.getModelFileDict')
def getModelFileDict(logger):

    try:

        modelDict = {
            'Xception'          : '../models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
            'VGG16'             : '../models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
            'VGG16_includeTop'  : '../models/vgg16_weights_tf_dim_ordering_tf_kernels.h5',
            'VGG19'             : '../models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            'InceptionV3'       : '../models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
            'MobileNet'         : '../models/mobilenet_1_0_224_tf_no_top.h5',
            'DenseNet'          : '../models/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
            'NASNet'            : '../models/nasnet_mobile_no_top.h5',
            'ResNet50'          : '../models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            'InceptionResNetV2' : '../models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
        }

        return modelDict

    except Exception as e:

        logger.error('Unable to get model file dictionary \n{}'.format(str(e)))

@lD.log(logBase + '.checkReady')
def checkReady(logger):

    try:

        modelString   = ['Xception', 'VGG16', 'VGG19', 'InceptionV3', 'MobileNet', 'DenseNet', 'NASNet', 
                         'ResNet50', 'InceptionResNetV2', 'VGG16_includeTop']
        modelDict     = getModelFileDict()

        for m in modelString:
            try:
                print('{} loading from {}...'.format(m, modelDict[m]), end='', flush=True)
                modelLoaded = modelImageNet(modelName=m, weightsFile=modelDict[m])
                print('sucessfully! '.format(m), end='', flush=True)
                print('type: {}'.format(type(modelLoaded)))
            except Exception as e:
                print('failed. --> {}'.format(m, str(e)))

    except Exception as e:

        logger.error('Unable to check ready \n{}'.format(str(e)))

@lD.log(logBase + '.main')
def main(logger, resultsDict):

    try:

        checkReady()

    except Exception as e:

        logger.error('Unable to run main \n{}'.format(str(e)))

if __name__ == '__main__':

    print('tf.__version__ :', tf.__version__)
    print('keras.__version__:', keras.__version__)