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
import config
from astropy.table import Table
import graphviz
import pydot
from keras.utils.vis_utils import plot_model
# import ML_Tensorflow.python.ML_Tensorflow

import tensorflow as tf
    
import logging

mydir = config.workpath('Test2')
table = Table.read(mydir + '/Measured_ksb.fits')

#features: e2 components missing
#features = np.array([table['aperture_sum'].T,table['sigma_mom'],table['rho4_mom'], table['sigma_mom_psf'], table['e1_cal'], table['e1_cal_psf']] )
Test = np.ones((33,10000,1))/20

Test_Preds = tf.ones((100,10000,1))*1.2
Test2 = tf.ones((33,10000,1))/10

# ncas = table.meta['N_CAS']
ncas = table.meta['N_CAS']
nreas = table.meta['N_REA']*table.meta['N_CANC']
feas = ['aperture_sum','sigma_mom', 'rho4_mom', 'sigma_mom_psf', 'e1_cal', 'e1_cal_psf']
ac = np.zeros((ncas, nreas,1))
e1 = np.zeros((ncas, nreas,1))
features = np.zeros((ncas, nreas,6))
fea_max = np.zeros(6)
count = 0
for i in feas:
    fea_max[count] = max(abs(table[i]))
    count = count + 1
for i in range(ncas):
    path = config.workpath('Test2/Measured_ksb_' + str(i) +'.fits')
    table =Table.read(path)
    for j in range(nreas):
        count_fea = 0
        ac[i][j][0] = table[j]['anisotropy_corr']
        e1[i][j][0] = table[j]['e1_cal']
        for k in feas:
            features[i][j][count_fea] = table[j][k]/fea_max[count_fea]
            count_fea = count_fea + 1
flags = []
for i in range(ncas):
    if (np.mean(features[i,:,0])*fea_max[0] < 500):
        flags.append(i)
features = np.delete(features, flags, 0)
ac = np.delete(ac, flags, 0)
e1 = np.delete(e1, flags, 0)
# ac = tf.convert_to_tensor(np.float32(ac))
# e1 = tf.convert_to_tensor(np.float32(e1))
# features = tf.convert_to_tensor(np.float32(features))
# print(len(features[1]))
# Test[0]= 0.1
# Test[0][1] = 0.2
# Test2[1]= 0.01
# print(Test_Preds)
# Target_Test = 0
Test_fea = tf.ones((33,10000,6))/5
# print(Test2)
def cost_func(preds, polarisation, anisotropy_corr):
    # preds = tf.convert_to_tensor(preds)
    # polarisation = tf.convert_to_tensor(polarisation)
    # anisotropy_corr = tf.convert_to_tensor(anisotropy_corr)
    if tf.keras.backend.ndim(preds) == 3:  
        cs = tf.keras.backend.mean(polarisation - preds * anisotropy_corr, axis = 1, keepdims = True)
        print(cs)
        cs_square = tf.keras.backend.square(cs)
        cs_square_mean = tf.keras.backend.mean(cs_square, keepdims = True) 
        return cs_square_mean
    else:
        print('false number of dimensions')

# print(cost_func(Test_Preds, e1, ac))


# nreas = 4
# input_fea=tf.keras.Input(8,dtype='float32') 
# input_fea = np.ones((1,8))
# targets = np.zeros((1,3,4))
# input_fea = tf.convert_to_tensor(input_fea)
# targets = tf.convert_to_tensor(targets)
# input_pointpreds = Test
# print(input_fea)

def mod(nreas = nreas, nfea = 6, hidden_layers = (3,3)):
    input_fea=tf.keras.Input((nreas, nfea),dtype='float32', name="Features") #feat
    # input_tar=tf.keras.Input((1, 1),dtype='float32', name="Targets") #targets
    # input_fea= [input_fea]
    auxilary_fea1=tf.keras.Input((nreas,1),dtype='float32', name="auxilary_features_1")
    auxilary_fea2=tf.keras.Input((nreas,1),dtype='float32', name="auxilary_features_2")
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
    model=(tf.keras.Model(inputs=[input_fea, auxilary_fea1, auxilary_fea2],outputs=outputs))
    model.add_loss(cost_func(x, auxilary_fea1, auxilary_fea2))
    return model
model = mod()
model.compile(
        loss = None,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), metrics = [])
history = model.fit([features, e1, ac], None, epochs = 100, verbose = 2, batch_size = None) 
print(model.predict(x = [features, e1, ac]))
tf.keras.utils.plot_model(model, to_file = 'Plots2/model.png', show_shapes=True)

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# hist.tail()

# model.summary()


# print(cost_func(Test_Preds, Test))
#print(tf.keras.backend.mean(tf.convert_to_tensor(Test[0]-Test_Preds*Test[1]), axis = 2 , keepdims = True))