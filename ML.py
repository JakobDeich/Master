import sys, os
# import ML_Tensorflow
import numpy as np
import random
import pickle
import matplotlib
import astropy
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from ML_Tensorflow.python.ML_Tensorflow import layer
from ML_Tensorflow.python.ML_Tensorflow import models
# import ML_Tensorflow.python.ML_Tensorflow

import tensorflow as tf
    
import logging


Test = tf.ones((3,4,1))/20

Test_Preds = tf.ones((3,4,1))/5
Test2 = tf.ones((3,4,1))/10
# Test[0]= 0.1
# Test[0][1] = 0.2
# Test2[1]= 0.01
# print(Test_Preds)
Target_Test = 0
Test_fea = tf.ones((3,4,8))/5
# print(Test2)
def cost_func(preds, polarisation, anisotropy_corr):
    # preds = tf.convert_to_tensor(preds)
    # polarisation = tf.convert_to_tensor(polarisation)
    # anisotropy_corr = tf.convert_to_tensor(anisotropy_corr)
    if tf.keras.backend.ndim(preds) == 3:  
        cs = tf.keras.backend.mean(polarisation - preds * anisotropy_corr, axis = 1, keepdims = True)
        cs_square = tf.keras.backend.square(cs)
        cs_square_mean = tf.keras.backend.mean(cs_square) 
        return cs_square_mean
    else:
        print('false number of dimensions')

# print(cost_func(Test_Preds, Test2, Test))


# nreas = 4
# input_fea=tf.keras.Input(8,dtype='float32') 
# input_fea = np.ones((1,8))
# targets = np.zeros((1,3,4))
# input_fea = tf.convert_to_tensor(input_fea)
# targets = tf.convert_to_tensor(targets)
# input_pointpreds = Test
# print(input_fea)

def mod(auxilary_fea1, auxilary_fea2, nreas = 4, nfea = 8, hidden_layers = (3,3)):
    input_fea=tf.keras.Input((nreas, nfea),dtype='float32', name="Features") #feat
    # input_tar=tf.keras.Input((1, 1),dtype='float32', name="Targets") #targets
    # input_fea= [input_fea]
    # auxilary_fea1=tf.keras.Input((nreas,1),dtype='float32', name="auxilary_features_1")
    # auxilary_fea2=tf.keras.Input((nreas,1),dtype='float32', name="auxilary_features_2")
    model = tf.keras.Sequential()
    # model.add(tf.keras.Input(shape=(nreas, nfea))) #reas and cases??
    for hidden_layer in hidden_layers:
        Layer = layer.TfbilacLayer(hidden_layer)
        model.add(Layer)
        model.add(tf.keras.layers.Activation('sigmoid'))
    model.add(layer.TfbilacLayer(1))
    x = model(input_fea)
    outputs = [x]
    # model.summary()
    model=(tf.keras.Model(inputs=input_fea,outputs=outputs))
    model.add_loss(cost_func(x, auxilary_fea1, auxilary_fea2))
    model.compile(
        loss = None,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
    return model

model = mod(Test2, Test)
# training_data = [Test2, Target_Test, Test]
history = model.fit(x = Test_fea, y = None, epochs = 100, verbose = 0) 

hist = pd.DataFrame(history.history)
print(hist)
# hist['epoch'] = history.epoch
# hist.tail()

# model.summary()


# print(cost_func(Test_Preds, Test))
#print(tf.keras.backend.mean(tf.convert_to_tensor(Test[0]-Test_Preds*Test[1]), axis = 2 , keepdims = True))