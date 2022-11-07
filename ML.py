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


Test = np.zeros((2,3,4))

Test_Preds = np.ones((1,3,4))

Test[0]= 0.1
Test[0][1] = 0.2
Test[1]= 0.01
# print(Test_Preds)
Target_Test = 0
Test2 = tf.ones((3,4,8))
#point_preds[0]: Galaxy polarisation
#point_preds[1]: anisotropy correction 
def cost_func(preds, targets, point_preds):
    preds = tf.convert_to_tensor(preds)
    point_preds = tf.convert_to_tensor(point_preds)
    if tf.keras.backend.ndim(preds) == 3:    
        cs = tf.keras.backend.mean(point_preds[0] - preds * point_preds[1], axis = 2, keepdims = True)
        cs_square = tf.keras.backend.square(cs)
        cs_square_mean = tf.keras.backend.mean(cs_square, keepdims = True) - targets
        return cs_square_mean
    else:
        print('false number of dimensions')

# print(cost_func(Test_Preds, Target_Test, Test))


nreas = 4
# input_fea=tf.keras.Input(8,dtype='float32') 
# input_fea = np.ones((1,8))
# targets = np.zeros((1,3,4))
# input_fea = tf.convert_to_tensor(input_fea)
# targets = tf.convert_to_tensor(targets)
# input_pointpreds = Test
# print(input_fea)
tf.compat.v1.enable_eager_execution()

def mod(Input, nreas = 4, ncas = 3, nfea = 8, hidden_layers = (3,3)):
    input_fea=tf.keras.Input((ncas, nreas, nfea),dtype='float32', name="Features") #feat
    input_tar=tf.keras.Input((1, 1),dtype='float32', name="Targets") #targets
    inputs= [input_fea,input_tar]
    input_pointpreds=tf.keras.Input((2, ncas, nreas),dtype='float32', name="Point_preds")
    inputs.append(input_pointpreds)
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(nfea,))) #reas and cases??
    for hidden_layer in hidden_layers:
        # model.add(tf.keras.layers.Dense(hidden_layer))
        Layer = layer.TfbilacLayer(hidden_layer)
        y = Layer(?) #what is the input here?
        # print(y)
        model.add(Layer)
        model.add(tf.keras.layers.Activation('sigmoid'))
        print('Hallo')
    model.add(tf.keras.layers.Dense(1))
    # model.summary()
    x = model(inputs[0])
    loss_func = cost_func(x, inputs[1], inputs[2])
    outputs = [x]
    model=(tf.keras.Model(inputs=inputs,outputs=outputs))
    model.add_loss(loss_func)
    return model

model = mod(Test2)
model.compile(
    loss = None,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
training_data = [Test2, Target_Test, Test]
# history = model.fit(x = input_fea, epochs = 100, verbose = 0) 

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# hist.tail()

# model.summary()


# print(cost_func(Test_Preds, Test))
#print(tf.keras.backend.mean(tf.convert_to_tensor(Test[0]-Test_Preds*Test[1]), axis = 2 , keepdims = True))