import sys, os
import numpy as np
import random
import astropy
import matplotlib
import tkinter 
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
from ML_Tensorflow.python.ML_Tensorflow import layer
# from ML_Tensorflow.python.ML_Tensorflow import models
import config
from astropy.table import Table
# from keras.utils.vis_utils import plot_model
# import ML_Tensorflow.python.ML_Tensorflow
matplotlib.use('Tkagg')
import tensorflow as tf
from astropy.table import Table
import logging
from pathlib import Path
import pickle

# aper = np.ones(30)*400
# sigma_mom = np.ones(30)*10
# ac = np.ones(30)*0.01
# e1 = np.ones(30)*0.01
# Pg = np.ones(30)
# for i in range(30):
#     if i >= 20:
#         ac[i] = 0.01/1.2
#         sigma_mom[i] = 30
#     elif i >= 10:
#         ac[i] = 0.01/1.1
#         sigma_mom[i] = 20
# for i in range(30):
#     if i%2 == 0:
#         ac[i] = 1 * ac[i]
#         e1[i] = 1 * e1[i]
# table1 = Table([aper, sigma_mom, e1], names = ['aperture_sum', 'sigma_mom', 'e1_cal'], dtype = ['f4', 'f4', 'f4'], meta = {'n_cas':3, 'n_rea': 10, 'n_canc':1})
# if not os.path.isdir(config.workpath('Test4')):
#     os.mkdir(config.workpath('Test4'))
# table1.write( config.workpath('Test4/Measured_ksb.fits') , overwrite=True) 
# table2 = Table([aper[0:10], sigma_mom[0:10], e1[0:10], ac[0:10], Pg[0:10]], names = ['aperture_sum', 'sigma_mom', 'e1_cal', 'anisotropy_corr', 'Pg_11'], dtype = ['f4', 'f4', 'f4', 'f4', 'f4'])
# table2.write( config.workpath('Test4/Measured_ksb_0.fits') , overwrite=True) 
# table2 = Table([aper[10:20], sigma_mom[10:20], e1[10:20], ac[10:20], Pg[10:20]], names = ['aperture_sum', 'sigma_mom', 'e1_cal', 'anisotropy_corr', 'Pg_11'], dtype = ['f4', 'f4', 'f4', 'f4', 'f4'])
# table2.write( config.workpath('Test4/Measured_ksb_1.fits') , overwrite=True) 
# table2 = Table([aper[20:30], sigma_mom[20:30], e1[20:30], ac[20:30], Pg[20:30]], names = ['aperture_sum', 'sigma_mom', 'e1_cal', 'anisotropy_corr', 'Pg_11'], dtype = ['f4', 'f4', 'f4', 'f4', 'f4'])
# table2.write( config.workpath('Test4/Measured_ksb_2.fits') , overwrite=True) 
# mydir = config.workpath('Test4')
# table = Table.read(mydir + '/Measured_ksb_2.fits')
# print(table)


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

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

def preprocessing(path, include_validation, val):
    print('start filtering')
    mydir = config.workpath(path)
    table = Table.read(mydir + '/Measured_ksb_1.fits')
    ncas = table.meta['N_CAS']
    fine_cases = []
    for i in range(ncas):
        path2 = config.workpath(path + '/Measured_ksb_' + str(i) +'.fits')
        file_name = Path(path2)
        if file_name.exists():
            table1 =Table.read(path2)
            if min(table1['e1_cal']) > -9:
                fine_cases.append(i)
    if include_validation == True:
        fine_cases_train, fine_cases_val = split_list(fine_cases)
        if val ==True:
            print('finish filtering with length: ' + str(len(fine_cases_val)))
            return fine_cases_val
        else:
            print('finish filtering with length: ' + str(len(fine_cases_train)))
            return fine_cases_train
    else:
        print('finish filtering with length: ' + str(len(fine_cases)))
        return fine_cases
            
            
    

def Data_processing(path, include_validation = False, val = False):
    print('start preprocessing')
    dic = {}
    mydir = config.workpath(path)
    table = Table.read(mydir + '/Measured_ksb_1.fits')
    ncas = table.meta['N_CAS']
    Training = preprocessing(path, include_validation, val)
    ncas = len(Training)
    nreas = table.meta['N_REA']*table.meta['N_CANC']
    feas = ['aperture_sum','sigma_mom', 'rho4_mom', 'sigma_mom_psf', 'e1_cal', 'e1_cal_psf', 'e2_cal', 'e2_cal_psf', 'anisotropy_corr', 'anisotropy_corr_2']
    #feas = ['aperture_sum','sigma_mom']
    ac = np.zeros((ncas, nreas,1))
    e1 = np.zeros((ncas, nreas,1))
    ac2 = np.zeros((ncas, nreas,1))
    e2 = np.zeros((ncas, nreas,1))
    # Pg = np.zeros((ncas, nreas,1))
    features = np.zeros((ncas, nreas,len(feas)))
    fea_max = np.zeros(len(feas))
    for count,i in enumerate(feas):
        fea_max[count] = max(abs(table[i]))
    for numb, value in enumerate(Training):
        path2 = config.workpath(path + '/Measured_ksb_' + str(value) +'.fits')
        table =Table.read(path2)
        for j in range(nreas):
            ac2[numb][j][0] = table[j]['anisotropy_corr_2']
            e2[numb][j][0] = table[j]['e2_cal']
            ac[numb][j][0] = table[j]['anisotropy_corr']
            e1[numb][j][0] = table[j]['e1_cal']
            # Pg[numb][j][0] = table[j]['Pg_11']
            for count_fea,k in enumerate(feas):
                features[numb][j][count_fea] = table[j][k]/fea_max[count_fea]
    flags = []
    for i in range(ncas):
        if (np.mean(features[i,:,0])*fea_max[0] < 0):
            flags.append(i)
    features = np.delete(features, flags, 0)
    ac = np.delete(ac, flags, 0)
    e1 = np.delete(e1, flags, 0)
    dic['features'] = features
    dic['anisotropy_corr'] = ac
    dic['e1_cal'] = e1
    dic['anisotropy_corr_2'] = ac2
    dic['e2_cal'] = e2
    # dic['Pg_11'] = Pg
    dic['n_rea'] = nreas
    dic['n_fea'] = len(feas)
    dic['n_cas'] = ncas
    pickle.dump( dic, open( mydir + "/Processed_data.p", "wb" ) )
    print('finished preprocessing')
    return dic

# dic = Data_processing('Test2', 1, True)
def cost_func(preds,  polarisation, anisotropy_corr, polarisation2,  anisotropy_corr2):
    preds = tf.convert_to_tensor(preds, dtype=float)
    preds1,preds2 = tf.split(preds, num_or_size_splits=2, axis = 2)
    polarisation = tf.convert_to_tensor(polarisation, dtype=float)
    anisotropy_corr = tf.convert_to_tensor(anisotropy_corr, dtype=float)
    polarisation2 = tf.convert_to_tensor(polarisation, dtype=float)
    anisotropy_corr2 = tf.convert_to_tensor(anisotropy_corr, dtype=float)
    if tf.keras.backend.ndim(preds) == 3:  
        # print(Minus_sq)
        # print(tf.keras.backend.mean(Minus_sq, axis = 1, keepdims = True))
        cs = tf.keras.backend.mean(polarisation - preds1 * anisotropy_corr, axis = 1, keepdims = True)
        cs_square = tf.keras.backend.square(cs)
        cs2 = tf.keras.backend.mean(polarisation2 - preds2 * anisotropy_corr2, axis = 1, keepdims = True)
        cs_square2 = tf.keras.backend.square(cs2)
        cs_square_mean = tf.keras.backend.mean(cs_square, keepdims = True)
        cs_square_mean2 = tf.keras.backend.mean(cs_square2, keepdims = True) 
        return cs_square_mean + cs_square_mean2
    else:
        print('false number of dimensions')

# preds = np.ones((99, 204000,2))
# pol = np.ones((99, 204000,1))*0.1
# pol2 = np.ones((99, 204000,1))*0.2
# ac = np.ones((99, 204000,1))*0.01
# ac2 = np.ones((99, 204000,1))*0.005
# print(cost_func(preds, pol, pol2, ac, ac2))


def create_model(nreas, nfea, hidden_layers = (3,3)):
    input_fea=tf.keras.Input((nreas, nfea),dtype='float32', name="Features") #feat
    auxilary_fea1=tf.keras.Input((nreas,1),dtype='float32', name="auxilary_features_1") #polarisation
    auxilary_fea2=tf.keras.Input((nreas,1),dtype='float32', name="auxilary_features_2") #anisotropy correction
    auxilary_fea3=tf.keras.Input((nreas,1),dtype='float32', name="auxilary_features_3") #polarisation second comp
    auxilary_fea4=tf.keras.Input((nreas,1),dtype='float32', name="auxilary_features_4") #anisotropy correction second comp
    model = tf.keras.Sequential()
    for hidden_layer in hidden_layers:
        Layer = layer.TfbilacLayer(hidden_layer)
        model.add(Layer)
        model.add(tf.keras.layers.Activation('sigmoid'))
    model.add(layer.TfbilacLayer(2))
    x = model(input_fea)
    outputs = [x]
    model=(tf.keras.Model(inputs=[input_fea, auxilary_fea1, auxilary_fea2, auxilary_fea3, auxilary_fea4],outputs=outputs))
    model.add_loss(cost_func(x, auxilary_fea1, auxilary_fea2, auxilary_fea3, auxilary_fea4))
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


# print(cost_func(tf.ones((3,10,1)), e1, ac))
# print(features)
checkpoint_path = config.workpath("training_1/cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)
def train(dic, checkpoint_path, epochs):
    model = create_model(dic['n_rea'], dic['n_fea'])
    model.compile(
            loss = None,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            metrics = [])
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=0,
            save_freq='epoch')
    model.save_weights(checkpoint_path.format(epoch=0))
    history = model.fit(
            [dic['features'], dic['e1_cal'], dic['anisotropy_corr'], dic['e2_cal'], dic['anisotropy_corr_2']], 
            None, 
            epochs = epochs,
            verbose = 2, 
            batch_size = None,
            callbacks=[cp_callback]) 
    return model, history

def validate(dic, checkpoint_path):
    checkpoint_path = config.workpath("training_1/cp.ckpt")
    model = create_model(dic['n_rea'], dic['n_fea'])
    model.compile(
            loss = None,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.06),
            metrics = [])
    model.load_weights(checkpoint_path)
    loss = model.evaluate([dic['features'], dic['e1_cal'], dic['anisotropy_corr'], dic['e2_cal'], dic['anisotropy_corr_2']])
    val_preds = model.predict(x = [dic['features'], dic['e1_cal'], dic['anisotropy_corr'], dic['e2_cal'], dic['anisotropy_corr_2']])
    return val_preds

def test(features_test, polarisation_test, anisotropy_corr_test, nreas, nfea, checkpoint_path):
    checkpoint_path = config.workpath("training_1/cp.ckpt")
    model = create_model(nreas, nfea)
    model.load_weights(checkpoint_path)
    test_preds = model.predict(x = [features_test, polarisation_test, anisotropy_corr_test])
    return test_preds



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
    aper_sum, aper_sum2 = oneD_to_threeD(path, param, dic['n_cas'], dic['n_rea'], 0.5)
    dic1 = mean_of(round(dic['n_cas']*case_percentage), e, e2, aper_sum)
    dic2 = mean_of(dic['n_cas'] - round(dic['n_cas']*case_percentage), e3, e4, aper_sum2)
    improvement = np.absolute(dic1['e_mean']/dic1['e2_mean'])
    improvement2 = np.absolute(dic2['e_mean']/dic2['e2_mean'])
    improvement_color = []
    if training_option:
        for i in improvement:
            if i > 1:
                improvement_color.append('red')
            else:
                improvement_color.append('blue')     
        plt.bar(dic1['param_mean'], improvement, width = max(dic1['param_mean'])/(round(dic['n_cas']*case_percentage)*3), align='center', color = improvement_color)
        plt.ylabel('fraction of non boosted PSF-anisotropy-corrected polarisations')
        plt.xlabel(param_name)
        plt.show()
    else:
        for i in improvement2:
            if i > 1:
                improvement_color.append('red')
            else:
                improvement_color.append('blue')
        plt.bar(dic2['param_mean'], improvement2, width = max(dic2['param_mean'])/(round(dic['n_cas']*case_percentage)*3),align='center', color = improvement_color)
        plt.ylabel('fraction of non boosted PSF-anisotropy-corrected polarisations')
        plt.xlabel(param_name)
        plt.show()
        

def cases_scatter_plot(dic,bsm, path, param, label,xlabel, first_component = True, non_boost=True):
    if first_component == True:
        e = (dic['e1_cal'] -  bsm * dic['anisotropy_corr'])
        e2 = (dic['e1_cal'] -  dic['anisotropy_corr'])
    else:
        e = (dic['e2_cal'] -  bsm * dic['anisotropy_corr_2'])
        e2 = (dic['e2_cal'] -  dic['anisotropy_corr_2'])        
    params = oneD_to_threeD(path, param, dic['n_cas'], dic['n_rea'])
    error = []
    for j in range(100):
        means = []
        mean = np.mean(e[j,:,0])
        for i in range(int(dic['n_rea']/120)):
            means.append(np.mean(e[j,i*120:(i+1)*120,0]))
        stabw = np.sum((means - mean)**2)/int(dic['n_rea']/120)
        stabw = np.sqrt(stabw)/np.sqrt(int(dic['n_rea']/120))
        error.append(stabw)   
    error2 = []
    for j in range(100):
        means2 = []
        mean2 = np.mean(e2[j,:,0])
        for i in range(int(dic['n_rea']/120)):
            means2.append(np.mean(e2[j,i*120:(i+1)*120,0]))
        stabw = np.sum((means2 - mean2)**2)/int(dic['n_rea']/120)
        stabw = np.sqrt(stabw)/np.sqrt(int(dic['n_rea']/120))
        error2.append(stabw)       
    dic1 = mean_of(dic['n_cas'], e, e2, params)
    if non_boost == True:
        plt.scatter(dic1['param_mean'], dic1['e2_mean'], marker = '.', color = 'r', label = '$b^{sm} = 1$')
        print(np.mean(np.square(dic1['e2_mean'])))
        plt.errorbar(dic1['param_mean'], dic1['e2_mean'], yerr = error2, ecolor = 'r', linestyle = ' ')
        plt.xlabel('$e_1^{PSF}$ psf polarisation')
        plt.ylabel('$c_1$ additive bias')
        plt.legend(loc = 'upper center')
        # plt.ylim([-0.0025, 0.0025])
        plt.tight_layout()
        # plt.savefig('Plots2/C-bias2/training_example0.png')
    # plt.scatter(dic1['param_mean'], dic1['e2_mean'], label = '$b^{sm} = 1$')
    plt.scatter(dic1['param_mean'], dic1['e_mean'], marker = '.',color = 'g', label = label)
    print(np.mean(np.square(dic1['e_mean'])))
    plt.errorbar(dic1['param_mean'], dic1['e_mean'], yerr = error, ecolor = 'green', linestyle = ' ')
    param_min = min(dic1['param_mean'])
    param_max= max(dic1['param_mean'])
    plt.axhline(y = 0, color = 'black', linestyle = 'dashed')
    # plt.xlabel('$e_1^{PSF}$ psf polarisation')
    plt.xlabel(xlabel)
    # plt.xlabel('Standard deviation of additive bias estimate')
    if first_component == True:
        plt.ylabel('$c_1$ additive bias')
    else:
        plt.ylabel('$c_2$ additive bias')
    plt.tight_layout()
    plt.legend(loc = 'upper center')
    # plt.savefig('Plots2/C-bias2/training_example.png')
    return None



# dic1 = mean_of(200, e, e2, aper_sum)
# print(np.mean(abs(dic1['e_mean'])), np.mean(abs(dic1['e2_mean'])))
# dic2 = mean_of(75 - round(75*case_percentage), e3, e4, aper_sum2)
# print(np.mean(abs(dic2['e_mean'])), np.mean(abs(dic2['e2_mean'])))
# bar = [np.mean(abs(dic1['e_mean'])), np.mean(abs(dic1['e2_mean'])), np.mean(abs(dic2['e_mean'])), np.mean(abs(dic2['e2_mean']))]
# y_pos = np.arange(len(bar))
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
    
    
# dic6 = Data_processing('Test4', True, True, False)
# model, history = train(dic6, checkpoint_path, 3500)
# dic7 = Data_processing('Test4', True, True, True)
# val_preds = validate(dic7, checkpoint_path)
# bsm = model.predict(x = [dic6['features'], dic6['e1_cal'], dic6['anisotropy_corr']])
# e = dic6['e1_cal']- bsm *dic6['anisotropy_corr']
# e_no = dic6['e1_cal']- dic6['anisotropy_corr']
# dic5 = mean_of(dic6['n_cas'], e, e_no, e)
# e = dic7['e1_cal']- val_preds *dic7['anisotropy_corr']
# e_no = dic7['e1_cal']- dic7['anisotropy_corr']
# dic2 = mean_of(dic7['n_cas'], e, e_no, e)
# dic = Data_processing('Test2', True, True, False)
# model2, history2 = train(dic, checkpoint_path, 3500)
# dic8 = Data_processing('Test2', True, True, True)
# val_preds2 = validate(dic8, checkpoint_path)
# bsm2 = model2.predict(x = [dic['features'], dic['e1_cal'], dic['anisotropy_corr']])
# e = dic['e1_cal']- bsm2 *dic['anisotropy_corr']
# e_no = dic['e1_cal']- dic['anisotropy_corr']
# dic1 = mean_of(dic['n_cas'], e, e_no, e)
# e = dic8['e1_cal']- val_preds2 *dic8['anisotropy_corr']
# e_no = dic8['e1_cal']- dic8['anisotropy_corr']
# dic3 = mean_of(dic8['n_cas'], e, e_no, e)


# bar = [np.mean(np.square(dic5['e_mean'])), np.mean(np.square(dic5['e2_mean'])),np.mean(np.square(dic2['e_mean'])), np.mean(np.square(dic2['e2_mean'])),np.mean(np.square(dic1['e_mean'])), np.mean(np.square(dic1['e2_mean'])), np.mean(np.square(dic3['e_mean'])), np.mean(np.square(dic3['e2_mean']))]
# y_pos = np.arange(len(bar))
# plt.bar(y_pos, bar, align='center', color = ['b', 'b', 'r', 'r', 'g', 'g'])
# #bar_name = ['trained $b_{sm}$ smaller sample', '$b_{sm} = 1$ with training data smaller sample','trained $b_{sm}$', '$b_{sm} = 1$ with training data', 'validated $b_{sm}$', '$b_{sm} = 1$ with validation data']
# bar_name = ['trained $b_{sm}$ \n more rotation', 'trained $b_{sm} = 1$ \n more rotation','validated $b_{sm}$ \n more rotation', 'validated $b_{sm} = 1$ \n more rotation','trained $b_{sm}$', '$b_{sm} = 1$', 'validated \n $b_{sm}$', 'validated \n $b_{sm} = 1$']
# plt.xticks(y_pos, bar_name)
# plt.ylabel('loss function')
# plt.show()
# plt.savefig('Plots2/C-bias2/more_rotation_epochs_validation3.png')

# cases_scatter_plot(dic2, bsm_val, 'Test2', 'e1_cal_psf','validation set estimate')
#boostFDep('sigma_mom', 'sigma moments')
#boostFDep('rho4_mom', 'rho4 moments')
# boostFDep('aperture_sum', 'aperture sum')
# print(e/P_g, e2/P_g)
# print(model.evaluate(x = [features, e1, ac], y = None))
# print(bsm)

# dic6 = Data_processing('Test5')
path = config.workpath('Test5')
dic6 = pickle.load( open(path + "/Processed_data.p", "rb" ) )
# # # bsm = np.ones((100,207000,1))
#model, history = train(dic6, checkpoint_path, 1500)
#bsm = model.predict(x = [dic6['features'], dic6['e1_cal'], dic6['anisotropy_corr'], dic6['e2_cal'], dic6['anisotropy_corr_2']])
path = config.workpath('Test5')
# file_name = os.path.join(path, 'BSM_test2.fits')
# t = Table([bsm], names = ['bsm'], dtype =['f4'])
# t.write(file_name, overwrite = True)
# bsm, bsm1 = np.split(bsm, indices_or_sections = 2,axis = 2)
table = Table.read(path + '/BSM_test2.fits')
bsm = table['bsm']
bsm, bsm1 = np.split(bsm, indices_or_sections = 2,axis = 2)
# plt.scatter(bsm, bsm1, marker = '.')
# plt.xlabel('$b^{sm}_1$')
# plt.ylabel('$b^{sm}_2$')
cases_scatter_plot(dic6, bsm, 'Test5', 'rho4_mom', 'ML boost-factor', 'sersic index estimate', True)
# model, history = train(dic6, checkpoint_path, 500)
# # # dic7 = Data_processing('Test2')
# # # val_preds = validate(dic7, checkpoint_path)
# bsm = model.predict(x = [dic6['features'], dic6['e1_cal'], dic6['anisotropy_corr']])
# path = config.workpath('Test3')
# file_name = os.path.join(path, 'BSM.fits')
# t = Table([bsm], names = ['bsm'], dtype =['f4'])
# t.write(file_name, overwrite = True)
# table = Table.read(path + '/BSM.fits')
# dic = mean_of(99, np.ones((99,96000,1)), np.ones((99,96000,1)), table['bsm'])
# dic2 = mean_of(99, np.ones((99,96000,1)), np.ones((99,96000,1)), dic6['features'][:,:,1])
# plt.scatter(dic2['param_mean'], dic['param_mean'], marker = '.')
# plt.ylabel('boost factor')
# plt.xlabel('galaxy size')
# bsm = 1
# cases_scatter_plot(dic6, bsm, 'Test3', 'e1_cal_psf','training set estimate')




# e = dic6['e1_cal']- bsm *dic6['anisotropy_corr']
# e_no = dic6['e1_cal']- dic6['anisotropy_corr']
# dic5 = mean_of(dic6['n_cas'], e, e_no, e)
# e = dic7['e1_cal']- val_preds *dic7['anisotropy_corr']
# e_no = dic7['e1_cal']- dic7['anisotropy_corr']
# dic8 = mean_of(dic7['n_cas'], e, e_no, e)
# print(np.mean(abs(dic5['e_mean']))/np.mean(abs(dic5['e2_mean'])))
# print(np.mean(abs(dic8['e_mean']))/np.mean(abs(dic8['e2_mean'])))
# cases_scatter_plot(dic6, 1, 'Test4', 'e1_cal_psf', 'training estimate')
# path = config.workpath('Comparison')
# table = Table.read(path + '/Measured_ksb.fits')
# # feas = ['aperture_sum','sigma_mom', 'rho4_mom', 'sigma_mom_psf', 'e1_cal', 'e1_cal_psf', 'Q111_psf', 'Q222_psf']
# feas = ['aperture_sum','sigma_mom', 'rho4_mom', 'sigma_mom_psf', 'e1_cal', 'e1_cal_psf']
# feats = np.zeros((table.meta['N_CAS'],1,len(feas)))
# e1 = np.zeros((table.meta['N_CAS'],1,1))
# ac = np.zeros((table.meta['N_CAS'],1,1))
# fea_max = np.zeros(len(feas))
# for count,i in enumerate(feas):
#     fea_max[count] = max(abs(table[i]))
# for i, Galaxy in enumerate(table):
#     e1[i] = Galaxy['e1_cal']
#     ac[i] = Galaxy['anisotropy_corr']
#     for j,k in enumerate(feas):
#         feats[i,0,j] = Galaxy[k]/fea_max[j]
# bsm = test(feats, e1, ac, 1 , len(feas), checkpoint_path)

# bar = [np.mean(abs(e1-bsm*ac)), np.mean(abs(e1-ac))]
# y_pos = np.arange(len(bar))
# plt.bar(y_pos, bar, align='center', color = ['b', 'r'])
# #bar_name = ['trained $b_{sm}$ smaller sample', '$b_{sm} = 1$ with training data smaller sample','trained $b_{sm}$', '$b_{sm} = 1$ with training data', 'validated $b_{sm}$', '$b_{sm} = 1$ with validation data']
# bar_name = ['ML $b_{sm}$', '$b_{sm} = 1$']
# plt.xticks(y_pos, bar_name)
# plt.ylabel('c-bias estimate')
# plt.show()
# plt.savefig('Plots2/C-bias2/testing.png')
# print(np.mean((e1-bsm*ac)**2), np.mean((e1-ac)**2))
# 0.07293662008416374

# plt.plot(range(750,1500), (history.history['loss'][750:1500]), label = 'big data')
# plt.plot(range(750,1500), (history2.history['loss'][750:1500]), label = 'small data')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()
# plt.show()
# plt.savefig('Plots2/C-bias2/loss_3.png')
# print(tf.keras.backend.ndim(bsm))
# print(cost_func(bsm, e1, ac))
# tf.keras.utils.plot_model(model, to_file = 'Plots2/model.png', show_shapes=True)

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# hist.tail()

# model.summary()


# print(cost_func(Test_Preds, Test))
#print(tf.keras.backend.mean(tf.convert_to_tensor(Test[0]-Test_Preds*Test[1]), axis = 2 , keepdims = True))