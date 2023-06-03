from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape


def Conv(n_filters, filter_width):
    return Conv2D(n_filters, filter_width, 
                  strides=2, padding="same", activation="relu")

def Deconv(n_filters, filter_width):
    return Conv2DTranspose(n_filters, filter_width, 
                           strides=2, padding="same", activation="relu")

def Encoder(inputs):
    X = Conv(32, 5)(inputs)
    X = Conv(64, 5)(X)
    X = Conv(128, 3)(X)
    X = Conv(256, 3)(X)
    X = Flatten()(X)
    Z = Dense(1024, activation="relu", name="encoder_output")(X)
    return Z

def Decoder(Z):
    X = Reshape((2, 2, 256), name="decoder_input")(Z)
    X = Deconv(128, 3)(X)
    X = Deconv(64, 5)(X)
    X = Deconv(32, 5)(X)
    X = Deconv(16, 5)(X)
    
    X = Deconv(8, 5)(X) # I add one Deconv(8,5) to reshape the output (64,64) to (128,128) 
    X = Deconv(16, 5)(X) # I change the last Deconv to fix the problem of "non-convergence" (see in class)
    
    X = Conv2D(1, 1)(X)
    return X 

def AutoEncoder():
    X = tf.keras.Input(shape=(32, 32, 1))
    Z = Encoder(X)
    X_pred = Decoder(Z)
    return tf.keras.Model(inputs=X, outputs=X_pred)

