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
logBase     = config['logging']['logBase'] + '.lib.model.autoencoder'

### turn off tensorflow info/warning/error or all python warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

class AE_simple:

    @lD.log(logBase + '.__init__')
    def __init__(logger, self, layers, activations, latentDim):

        self.layers      = layers
        self.activations = activations
        self.latentDim   = latentDim

    @lD.log(logBase + '.__call__')
    def __call__(logger, self, inputs):

        try:

            prevLayer = inputs

            ### Encoder network
            with tf.variable_scope('Encoder_Network'):
                for index, (nodes, activation) in enumerate( zip( self.layers, self.activations )):
                    denseLayer  = tf.layers.Dense(units=nodes, activation=activation, name='encodeLayer_{}'.format(index + 1))
                    prevLayer   = denseLayer( prevLayer )

            ### Latent space
            with tf.variable_scope('Latent'):
                latentLayer  = tf.layers.Dense(units=self.latentDim, activation=activation, name='latent_layer')
                prevLayer    = latentLayer(prevLayer)

            ### Decoder network
            with tf.variable_scope('Decoder_Network'):
                for index, (nodes, activation) in enumerate( zip( reversed(self.layers), reversed(self.activations) )):
                    denseLayer  = tf.layers.Dense(units=nodes, activation=activation, name='decodeLayer_{}'.format(index + 1))
                    prevLayer   = denseLayer( prevLayer )

            return prevLayer

        except Exception as e:

            logger.error('Unable to get autoencoder network \n{}'.format(str(e)))