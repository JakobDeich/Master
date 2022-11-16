import sys, os
# import ML_Tensorflow
import numpy as np
import random
import pickle
import matplotlib
import astropy
# matplotlib.use('Agg')
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

mydir = config.workpath('Test3')
table = Table.read(mydir + '/Measured_ksb.fits')

def threeD_to_oneD(array, ncas, nrea):
    tab = np.zeros(ncas*nrea)
    for case in range(ncas):
        for rea in range(nrea):
            tab[case * nrea + rea] = array[case][rea][0]       
    return tab

def oneD_to_threeD(path, parameter, ncas, nreas):
    array = np.zeros((ncas , nreas,1))
    for i in range(ncas):
        path2 = config.workpath(path + '/Measured_ksb_' + str(i) +'.fits')
        table =Table.read(path2)
        for j in range(nreas):
            array[i][j][0] = table[j][parameter]
    return array

def Data_processing(path, case_percentage = 1):
    mydir = config.workpath(path)
    table = Table.read(mydir + '/Measured_ksb.fits')
    ncas = table.meta['N_CAS']
    Training = np.arange(round(ncas * case_percentage))
    Testing = np.arange(round(ncas * case_percentage), ncas)
    nreas = table.meta['N_REA']*table.meta['N_CANC']
    feas = ['aperture_sum','sigma_mom', 'rho4_mom', 'sigma_mom_psf', 'e1_cal', 'e1_cal_psf']
    ac = np.zeros((round(ncas * case_percentage), nreas,1))
    e1 = np.zeros((round(ncas * case_percentage), nreas,1))
    Pg = np.zeros((round(ncas * case_percentage), nreas,1))
    ac_test = np.zeros((ncas - round(ncas * case_percentage), nreas,1))
    e1_test = np.zeros((ncas - round(ncas * case_percentage), nreas,1))
    Pg_test = np.zeros((ncas -round(ncas * case_percentage), nreas,1))
    features_test = np.zeros((ncas -round(ncas * case_percentage), nreas,6))
    features = np.zeros((round(ncas * case_percentage), nreas,6))
    fea_max = np.zeros(6)
    count = 0
    for i in feas:
        fea_max[count] = max(abs(table[i]))
        count = count + 1
    for i in Training:
        path2 = config.workpath(path + '/Measured_ksb_' + str(i) +'.fits')
        table =Table.read(path2)
        for j in range(nreas):
            count_fea = 0
            ac[i][j][0] = table[j]['anisotropy_corr']
            e1[i][j][0] = table[j]['e1_cal']
            Pg[i][j][0] = table[j]['Pg_11']
            for k in feas:
                features[i][j][count_fea] = table[j][k]/fea_max[count_fea]
                count_fea = count_fea + 1
    for i,m in enumerate(Testing):
        path2 = config.workpath(path + '/Measured_ksb_' + str(m) +'.fits')
        table =Table.read(path2)
        for j in range(nreas):
            ac_test[i][j][0] = table[j]['anisotropy_corr']
            e1_test[i][j][0] = table[j]['e1_cal']
            Pg_test[i][j][0] = table[j]['Pg_11']
            for k,l in enumerate(feas):
                features_test[i][j][k] = table[j][l]/fea_max[k]
    flags = []
    for i in range(round(ncas * case_percentage)):
        if (np.mean(features[i,:,0])*fea_max[0] < 0):
            flags.append(i)
    features = np.delete(features, flags, 0)
    ac = np.delete(ac, flags, 0)
    e1 = np.delete(e1, flags, 0)
    flags_test = []
    for i in range(ncas - round(ncas * case_percentage)):
        if (np.mean(features_test[i,:,0])*fea_max[0] < 0):
            flags_test.append(i)
    features_test = np.delete(features_test, flags, 0)
    ac_test = np.delete(ac_test, flags, 0)
    e1_test = np.delete(e1_test, flags, 0)
    return features, ac, e1, Pg, nreas, len(feas), features_test, ac_test, e1_test, Pg_test


def cost_func(preds, polarisation, anisotropy_corr):
    if tf.keras.backend.ndim(preds) == 3:  
        cs = tf.keras.backend.mean(polarisation - preds * anisotropy_corr, axis = 1, keepdims = True)
        cs_square = tf.keras.backend.square(cs)
        cs_square_mean = tf.keras.backend.mean(cs_square, keepdims = True) 
        return cs_square_mean
    else:
        print('false number of dimensions')


def mod(nreas, nfea, hidden_layers = (3,3)):
    input_fea=tf.keras.Input((nreas, nfea),dtype='float32', name="Features") #feat
    auxilary_fea1=tf.keras.Input((nreas,1),dtype='float32', name="auxilary_features_1") #polarisation
    auxilary_fea2=tf.keras.Input((nreas,1),dtype='float32', name="auxilary_features_2") #anisotropy correction
    model = tf.keras.Sequential()
    for hidden_layer in hidden_layers:
        Layer = layer.TfbilacLayer(hidden_layer)
        model.add(Layer)
        model.add(tf.keras.layers.Activation('sigmoid'))
    model.add(layer.TfbilacLayer(1))
    x = model(input_fea)
    outputs = [x]
    model=(tf.keras.Model(inputs=[input_fea, auxilary_fea1, auxilary_fea2],outputs=outputs))
    model.add_loss(cost_func(x, auxilary_fea1, auxilary_fea2))
    return model


features, ac, e1,Pg, nreas, nfea, features_test, ac_test, e1_test, Pg_test = Data_processing('Test3', 1)
model = mod(nreas, nfea)
model.compile(
    loss = None,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    metrics = [])
history = model.fit(
    [features, e1, ac], 
    None, 
    epochs = 100,
    verbose = 0, 
    batch_size = None) 
bsm = model.predict(x = [features, e1, ac])
# print(bsm)
# e = np.mean((e1 -  bsm * ac)**2)
# e2 = np.mean((e1 -  ac)**2)
# P_g = np.mean(Pg)
aper_sum = oneD_to_threeD('Test3', 'sigma_mom', 75, nreas)
bsm_mean = np.zeros((75,1))
aper_mean = np.zeros((75,1))    
points = 5
for i in range(75):
    bsm_mean[i] = np.mean(bsm[i,:])
    aper_mean[i] = np.mean(aper_sum[i,:])
list1 = np.linspace(min(aper_mean),max(aper_mean), points, endpoint = False)
bsms = []
step = max(aper_mean)/points
for i in range(points):
    Good1 = aper_mean > list1[i]
    Good2 = aper_mean < list1[i] + step
    Good = np.logical_and(Good1, Good2)
    bsms.append(np.mean(bsm_mean[Good]))
plt.scatter(list1,bsms)
plt.xlabel('sigma moments moments')
plt.ylabel('boost factor')
plt.savefig('Plots2/sigma_boost.pdf')
    
# print(e - e2)
# print(model.evaluate(x = [features, e1, ac], y = None))
# print(bsm)
# plt.plot(range(100), (history.history['loss']), label= str(i * 100) +' percent of cases')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# print(tf.keras.backend.ndim(bsm))
# print(cost_func(bsm, e1, ac))
# tf.keras.utils.plot_model(model, to_file = 'Plots2/model.png', show_shapes=True)

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# hist.tail()

# model.summary()


# print(cost_func(Test_Preds, Test))
#print(tf.keras.backend.mean(tf.convert_to_tensor(Test[0]-Test_Preds*Test[1]), axis = 2 , keepdims = True))