import sys, os
import numpy as np
import random
import astropy
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from ML_Tensorflow.python.ML_Tensorflow import layer
# from ML_Tensorflow.python.ML_Tensorflow import models
import config
from astropy.table import Table
# from keras.utils.vis_utils import plot_model
import tkinter 
# import ML_Tensorflow.python.ML_Tensorflow

import tensorflow as tf
    
import logging

aper = np.ones(30)*400
sigma_mom = np.ones(30)*10
ac = np.ones(30)*0.01
e1 = np.ones(30)*0.01
Pg = np.ones(30)
for i in range(30):
    if i >= 20:
        ac[i] = 0.01/1.2
        sigma_mom[i] = 30
    elif i >= 10:
        ac[i] = 0.01/1.1
        sigma_mom[i] = 20
for i in range(30):
    if i%2 == 0:
        ac[i] = 1 * ac[i]
        e1[i] = 1 * e1[i]
table1 = Table([aper, sigma_mom, e1], names = ['aperture_sum', 'sigma_mom', 'e1_cal'], dtype = ['f4', 'f4', 'f4'], meta = {'n_cas':3, 'n_rea': 10, 'n_canc':1})
if not os.path.isdir(config.workpath('Test4')):
    os.mkdir(config.workpath('Test4'))
table1.write( config.workpath('Test4/Measured_ksb.fits') , overwrite=True) 
table2 = Table([aper[0:10], sigma_mom[0:10], e1[0:10], ac[0:10], Pg[0:10]], names = ['aperture_sum', 'sigma_mom', 'e1_cal', 'anisotropy_corr', 'Pg_11'], dtype = ['f4', 'f4', 'f4', 'f4', 'f4'])
table2.write( config.workpath('Test4/Measured_ksb_0.fits') , overwrite=True) 
table2 = Table([aper[10:20], sigma_mom[10:20], e1[10:20], ac[10:20], Pg[10:20]], names = ['aperture_sum', 'sigma_mom', 'e1_cal', 'anisotropy_corr', 'Pg_11'], dtype = ['f4', 'f4', 'f4', 'f4', 'f4'])
table2.write( config.workpath('Test4/Measured_ksb_1.fits') , overwrite=True) 
table2 = Table([aper[20:30], sigma_mom[20:30], e1[20:30], ac[20:30], Pg[20:30]], names = ['aperture_sum', 'sigma_mom', 'e1_cal', 'anisotropy_corr', 'Pg_11'], dtype = ['f4', 'f4', 'f4', 'f4', 'f4'])
table2.write( config.workpath('Test4/Measured_ksb_2.fits') , overwrite=True) 
# mydir = config.workpath('Test4')
# table = Table.read(mydir + '/Measured_ksb_2.fits')
# print(table)


def threeD_to_oneD(array, ncas, nrea):
    tab = np.zeros(ncas*nrea)
    for case in range(ncas):
        for rea in range(nrea):
            tab[case * nrea + rea] = array[case][rea][0]       
    return tab

def oneD_to_threeD(path, parameter, ncas, nreas, case_percentage):
    array = np.zeros((round(ncas*case_percentage) , nreas,1))
    array2 = np.zeros((round(ncas - ncas*case_percentage) , nreas,1))
    for i in range(round(ncas*case_percentage)):
        path2 = config.workpath(path + '/Measured_ksb_' + str(i) +'.fits')
        table =Table.read(path2)
        for j in range(nreas):
            array[i][j][0] = table[j][parameter]
    for i,k in enumerate(np.arange(round(ncas*case_percentage), ncas)):
        path2 = config.workpath(path + '/Measured_ksb_' + str(k) +'.fits')
        table =Table.read(path2)
        for j in range(nreas):
            array2[i][j][0] = table[j][parameter]    
    return array, array2



def Data_processing(path, case_percentage = 1):
    dic = {}
    mydir = config.workpath(path)
    table = Table.read(mydir + '/Measured_ksb.fits')
    ncas = table.meta['N_CAS']
    Training = np.arange(round(ncas * case_percentage))
    Testing = np.arange(round(ncas * case_percentage), ncas)
    nreas = table.meta['N_REA']*table.meta['N_CANC']
    feas = ['aperture_sum','sigma_mom', 'rho4_mom', 'sigma_mom_psf', 'e1_cal', 'e1_cal_psf']
    #feas = ['aperture_sum','sigma_mom']
    ac = np.zeros((round(ncas * case_percentage), nreas,1))
    e1 = np.zeros((round(ncas * case_percentage), nreas,1))
    Pg = np.zeros((round(ncas * case_percentage), nreas,1))
    ac_test = np.zeros((ncas - round(ncas * case_percentage), nreas,1))
    e1_test = np.zeros((ncas - round(ncas * case_percentage), nreas,1))
    Pg_test = np.zeros((ncas -round(ncas * case_percentage), nreas,1))
    features_test = np.zeros((ncas -round(ncas * case_percentage), nreas,len(feas)))
    features = np.zeros((round(ncas * case_percentage), nreas,len(feas)))
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
    dic['features'] = features
    dic['anisotropy_corr'] = ac
    dic['e1_cal'] = e1
    dic['Pg_11'] = Pg
    dic['features_test'] = features_test
    dic['anisotropy_corr_test'] = ac_test
    dic['e1_cal_test'] = e1_test
    dic['Pg_11_test'] = Pg_test
    dic['n_rea'] = nreas
    dic['n_fea'] = len(feas)
    return dic


def cost_func(preds, polarisation, anisotropy_corr):
    preds = tf.convert_to_tensor(preds, dtype=float)
    polarisation = tf.convert_to_tensor(polarisation, dtype=float)
    anisotropy_corr = tf.convert_to_tensor(anisotropy_corr, dtype=float)
    if tf.keras.backend.ndim(preds) == 3:  
        # print(Minus_sq)
        # print(tf.keras.backend.mean(Minus_sq, axis = 1, keepdims = True))
        cs = tf.keras.backend.mean(polarisation - preds * anisotropy_corr, axis = 1, keepdims = True)
        cs_square = tf.keras.backend.square(cs)
        cs_square_mean = tf.keras.backend.mean(cs_square, keepdims = True) 
        return cs_square_mean
    else:
        print('false number of dimensions')


def create_model(nreas, nfea, hidden_layers = (3,3)):
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

def boostFDep(Parameter, bsm, nreas, Plot_Name):
    aper_sum = oneD_to_threeD('Test2', Parameter, 75, nreas)
    bsm_mean = np.zeros((75))
    aper_mean = np.zeros((75))    
    for i in range(75):
        bsm_mean[i] = np.mean(bsm[i,:])
        aper_mean[i] = np.mean(aper_sum[i,:])
    list2 = Table([aper_mean,bsm_mean], names = ('aper', 'bsm'), dtype = ['f4', 'f4'])
    list2.sort('aper')
    bsms = []
    apers = []
    step = round(len(list2)/5)
    for i in range(5):
        dummy = []
        dummy2 = []
        for j in range(step):
            dummy.append(list2['bsm'][i*step + j])
            dummy2.append(list2['aper'][i*step + j])    
        bsms.append(np.mean(dummy))
        apers.append(np.mean(dummy2))
    plt.scatter(apers,bsms)
    plt.xlabel(Plot_Name)
    plt.ylabel('boost factor')
    plt.show()

dic = Data_processing('Test3', 1)
# print(cost_func(tf.ones((3,10,1)), e1, ac))
# print(features)
checkpoint_path = config.workpath("training_1/cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)
def train(dic, checkpoint_path):
    model = create_model(dic['n_rea'], dic['n_fea'])
    model.compile(
            loss = None,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            metrics = [])
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=1)
    history = model.fit(
            [dic['features'], dic['e1_cal'], dic['anisotropy_corr']], 
            None, 
            epochs = 100,
            verbose = 2, 
            batch_size = None,
            callbacks=[cp_callback]) 
    return model, history

def validate(dic, checkpoint_path):
    checkpoint_path = config.workpath("training_1/cp.ckpt")
    model = create_model(dic['n_rea'], dic['n_fea'])
    model.compile(
            loss = None,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            metrics = [])
    model.load_weights(checkpoint_path)
    loss = model.evaluate([dic['features_test'], dic['e1_cal_test'], dic['anisotropy_corr_test']])
    print(loss)
    val_preds = model.predict(x = [dic['features_test'], dic['e1_cal_test'], dic['anisotropy_corr_test']])
    return val_preds

def test(features_test, polarisation_test, anisotropy_corr_test, nreas, nfea, checkpoint_path):
    checkpoint_path = config.workpath("training_1/cp.ckpt")
    model = create_model(nreas, nfea)
    model.load_weights(checkpoint_path)
    test_preds = model.predict(x = [features_test, polarisation_test, anisotropy_corr_test])
    return test_preds
model, history = train(dic, checkpoint_path)
# val_preds = validate(dic, checkpoint_path)
bsm = model.predict(x = [dic['features'], dic['e1_cal'], dic['anisotropy_corr']])
# print(bsm)

def mean_of(ncas, e1, e1_no_boost, param):
    dic = {}
    e_mean = np.zeros(ncas)
    param_mean = np.zeros(ncas) 
    e2_mean = np.zeros(ncas)  
    for i in range(ncas):
        e_mean[i] = np.mean(e1[i,:])
        e2_mean[i] = np.mean(e1_no_boost[i,:])
        param_mean[i] = np.mean(param[i,:])
    dic['e_mean'] = e_mean
    dic['e2_mean'] = e2_mean
    dic['param_mean'] = param_mean
    return dic

def cases(dic, path, bsm, val_preds, case_percentage, param, param_name, training_option = True):
    e = (dic['e1_cal'] -  bsm * dic['anisotropy_corr'])
    e3 = (dic['e1_cal_test'] -  val_preds * dic['anisotropy_corr_test'])
    e2 = (dic['e1_cal'] -  dic['anisotropy_corr'])
    e4 = (dic['e1_cal_test'] -  dic['anisotropy_corr_test'])
    aper_sum, aper_sum2 = oneD_to_threeD(path, param, 75, dic['n_rea'], 0.5)
    dic1 = mean_of(round(75*case_percentage), e, e2, aper_sum)
    dic2 = mean_of(75 - round(75*case_percentage), e3, e4, aper_sum2)
    improvement = np.absolute(dic1['e_mean']/dic1['e2_mean'])
    improvement2 = np.absolute(dic2['e_mean']/dic2['e2_mean'])
    improvement_color = []
    if training_option:
        for i in improvement:
            if i > 1:
                improvement_color.append('red')
            else:
                improvement_color.append('blue')     
        plt.bar(dic1['param_mean'], improvement, width = max(dic1['param_mean'])/(round(75*case_percentage)*3), align='center', color = improvement_color)
        plt.ylabel('fraction of non boosted PSF-anisotropy-corrected polarisations')
        plt.xlabel(param_name)
        plt.show()
    else:
        for i in improvement2:
            if i > 1:
                improvement_color.append('red')
            else:
                improvement_color.append('blue')
        plt.bar(dic2['param_mean'], improvement2, width = max(dic2['param_mean'])/(round(75*case_percentage)*3),align='center', color = improvement_color)
        plt.ylabel('fraction of non boosted PSF-anisotropy-corrected polarisations')
        plt.xlabel(param_name)
        plt.show()
        

def cases_scatter_plot(dic,bsm, path, param):
    e = (dic['e1_cal'] -  bsm * dic['anisotropy_corr'])
    e2 = (dic['e1_cal'] -  dic['anisotropy_corr'])
    aper_sum, aper_sum2 = oneD_to_threeD(path, param, 75, dic['n_rea'], 1)
    dic1 = mean_of(75, e, e2, aper_sum)
    plt.scatter(dic1['param_mean'], dic1['e2_mean'], marker = '.', label = '$b^{sm} = 1$')
    plt.xlabel('psf polarisation')
    plt.ylabel('psf anisotropy corrected polarisation per case')
    plt.legend(loc = 'upper center')
    plt.ylim([-0.0025, 0.0025])
    plt.tight_layout()
    plt.savefig('Plots2/C-bias/Conf_noBSM.png')
    plt.show()
    # plt.scatter(dic1['param_mean'], dic1['e2_mean'], label = '$b^{sm} = 1$')
    plt.scatter(dic1['param_mean'], dic1['e_mean'], marker = '.', label = 'neural network estimate for $b^{sm}$')
    plt.xlabel('psf polarisation')
    plt.ylabel('psf anisotropy corrected polarisation per case')
    plt.legend(loc = 'upper center')
    plt.savefig('Plots2/C-bias/Conf_BSM.png')
    return None



    # dic1 = mean_of(round(75*case_percentage), e, e2, aper_sum)
    # print(np.mean(abs(dic1['e_mean'])), np.mean(abs(dic1['e2_mean'])))
    # dic2 = mean_of(75 - r'aperture sumound(75*case_percentage), e3, e4, aper_sum2)
    # print(np.mean(abs(dic2['e_mean'])), np.mean(abs(dic2['e2_mean'])))
    # bar = [np.mean(abs(dic1['e_mean'])), np.mean(abs(dic1['e2_mean'])), np.mean(abs(dic2['e_mean'])), np.mean(abs(dic2['e2_mean']))]
    # y_pos = np.arange(len(bar))'psf anisotropy corrected polarisation per case'
    # plt.bar(y_pos, bar, align='center', color = ['b', 'b', 'r', 'r'])
    # bar_name = ['trained $b_{sm}$', '$b_{sm} = 1$ with training data', 'validated $b_{sm}$', '$b_{sm} = 1$ with validation data']
    # plt.xticks(y_pos, bar_name)
    # plt.ylabel('Mean of the absolute value of the PSF-anisotropy-corrected polarisations of each case')
    # plt.scatter(dic1['param_mean'], dic1['e_mean'], marker = '.', label = '$b_{sm}$ training')
    # plt.scatter(dic1['param_mean'], dic1['e2_mean'], marker = '.', label = '$b_{sm} = 1$')
    # plt.scatter(dic2['param_mean'], dic2['e_mean'], marker = '.', label = '$b_{sm}$ validation')
    # plt.scatter(dic2['param_mean'], dic2['e2_mean'], marker = '.', label = '$b_{sm} = 1$ validation')
    # plt.xlabel(param_name)
    # plt.ylabel('Mean PSF-anisotropy-corrected polarisation')
    # plt.legend()
    # plt.show()
    
    
# cases(dic, 'Test3', bsm, val_preds, 0.5, 'aperture_sum', 'aperture sum', False)
cases_scatter_plot(dic, bsm, 'Test3', 'e1_cal_psf')
#boostFDep('sigma_mom', 'sigma moments')
#boostFDep('rho4_mom', 'rho4 moments')
# boostFDep('aperture_sum', 'aperture sum')
# print(e/P_g, e2/P_g)
# print(model.evaluate(x = [features, e1, ac], y = None))
# print(bsm)
# plt.plot(range(100), (history.history['loss']))
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()
# print(tf.keras.backend.ndim(bsm))
# print(cost_func(bsm, e1, ac))
# tf.keras.utils.plot_model(model, to_file = 'Plots2/model.png', show_shapes=True)

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# hist.tail()

# model.summary()


# print(cost_func(Test_Preds, Test))
#print(tf.keras.backend.mean(tf.convert_to_tensor(Test[0]-Test_Preds*Test[1]), axis = 2 , keepdims = True))