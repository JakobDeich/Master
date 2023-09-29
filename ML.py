import sys, os
import numpy as np
import random
import astropy
import matplotlib
# import tkinter 
# matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
from ML_Tensorflow.python.ML_Tensorflow import layer
# from ML_Tensorflow.python.ML_Tensorflow import models
import config
from astropy.table import Table
# from keras.utils.vis_utils import plot_model
# import ML_Tensorflow.python.ML_Tensorflow
# matplotlib.use('Tkagg')
import tensorflow as tf
from astropy.table import Table, vstack
import logging
from pathlib import Path
import pickle
from multiprocessing import Pool, cpu_count
#from matplotlib.ticker import SymmetricalLogLocator
# import analyze


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

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
    # mydir = config.workpath(path)
    # table = Table.read(mydir + '/Measured_ksb_1.fits')
    # ncas = table.meta['N_CAS']
    ncas = 100
    fine_cases = []
    for i in range(ncas):
        path2 = config.workpath(path + '/Measured_ksb_' + str(i) +'.fits')
        file_name = Path(path2)
        if file_name.exists():
            table1 =Table.read(path2)
            if min(table1['e1_cal']) > -9 and max(table1['mag']) <= 24.51 :
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
            
            
plt.rcParams.update({'font.size': 14}) 

def Data_processing(path, include_validation = False, val = False):
    print('start preprocessing')
    dic = {}
    mydir = config.workpath(path)
    # table = Table.read(mydir + '/Measured_ksb_1.fits')
    # ncas = table.meta['N_CAS']
    # ncas = 100
    Training = preprocessing(path, include_validation, val)
    ncas = len(Training)
    # nreas = table.meta['N_REA']*table.meta['N_CANC']
    nreas = 1700*120
    # if include_Ms == True:
    # 	feas = ['aperture_sum','sigma_mom', 'rho4_mom', 'sigma_mom_psf', 'e1_cal', 'e1_cal_psf', 'anisotropy_corr', 'M4_1_psf_adap']
    # 	feas2 = ['aperture_sum','sigma_mom', 'rho4_mom', 'sigma_mom_psf', 'e2_cal', 'e2_cal_psf','anisotropy_corr_2','M4_2_psf_adap']
    # 	fea_max = [2000, 40, 2.6, 40,0.7,0.11,0.02, 0.06] 
    # else:
    # 	feas = ['aperture_sum','sigma_mom', 'rho4_mom', 'sigma_mom_psf', 'e1_cal', 'e1_cal_psf', 'anisotropy_corr']
    # 	feas2 = ['aperture_sum','sigma_mom', 'rho4_mom', 'sigma_mom_psf', 'e2_cal', 'e2_cal_psf','anisotropy_corr_2']
    # 	fea_max = [2000, 40, 2.6, 40,0.7,0.11,0.02]
    #feas = ['aperture_sum','sigma_mom', 'rho4_mom', 'sigma_mom_psf', 'e1_cal', 'e2_cal', 'e1_cal_psf','e2_cal_psf' , 'anisotropy_corr','anisotropy_corr_2', 'M4_1_psf_adap', 'M4_2_psf_adap']
    feas = ['aperture_sum','sigma_mom', 'rho4_mom', 'sigma_mom_psf', 'e1_cal', 'e2_cal', 'e1_cal_psf','e2_cal_psf' , 'anisotropy_corr','anisotropy_corr_2']
    #fea_max= [2000, 40, 2.6, 40, 2.6,0.7, 0.7, 0.11, 0.11 ,0.02, 0.02, 0.06, 0.06] 
    fea_max= [2000, 40, 2.6, 40, 2.6,0.7, 0.7, 0.11, 0.11 ,0.02, 0.02] 
    ac = np.zeros((ncas, nreas,1))
    e1 = np.zeros((ncas, nreas,1))
    ac2 = np.zeros((ncas, nreas,1))
    e2 = np.zeros((ncas, nreas,1))
    # Pg = np.zeros((ncas, nreas,1))
    features = np.zeros((ncas, nreas,len(feas)))
    #features2 = np.zeros((ncas, nreas, len(feas2)))
    # for count,i in enumerate(feas):
    #     fea_max[count] = max(abs(table[i]))
    for numb, value in enumerate(Training):
        path2 = config.workpath(path + '/Measured_ksb_' + str(value) +'.fits')
        table =Table.read(path2)
        if numb%5==0:
            print(numb)
        for j in range(nreas):
            ac2[numb][j][0] = table[j]['anisotropy_corr_2']
            e2[numb][j][0] = table[j]['e2_cal']
            ac[numb][j][0] = table[j]['anisotropy_corr']
            e1[numb][j][0] = table[j]['e1_cal']
            # Pg[numb][j][0] = table[j]['Pg_11']
            for count_fea,k in enumerate(feas):
              features[numb][j][count_fea] = table[j][k]/fea_max[count_fea]
            # for count_fea,k in enumerate(feas2):
            #   features2[numb][j][count_fea] = table[j][k]/fea_max[count_fea]
    #flags = []
    #for i in range(ncas):
    #    if (np.mean(features[i,:,0])*fea_max[0] < 0):
    #        flags.append(i)
    #features = np.delete(features, flags, 0)
    #ac = np.delete(ac, flags, 0)
    #e1 = np.delete(e1, flags, 0)
    dic['features'] = features
    # dic['features_2'] = features2
    dic['anisotropy_corr'] = ac
    dic['e1_cal'] = e1
    dic['anisotropy_corr_2'] = ac2
    dic['e2_cal'] = e2
    # dic['Pg_11'] = Pg
    dic['n_rea'] = nreas
    dic['n_fea'] = len(feas)
    dic['n_cas'] = ncas
    if val == False:
        pickle.dump( dic, open( mydir + "/Processed_data_noM.p", "wb" ) )
    else:
        pickle.dump( dic, open( mydir + "/Processed_valdata_noM.p", "wb" ) )
    print('finished preprocessing')
    return dic

def Flags(path, dic, SN, flags, limit):
    for j in range(dic['n_cas']):
        # print(j)
        tab = Table.read(config.workpath(path) + '/Measured_ksb_' + str(j) + '.fits')
        if (np.mean(tab['Signal_to_Noise']) > limit):
            flags.append(j)
            SN.append(np.mean(tab['Signal_to_Noise']))
    return flags, SN

def Flagging(dic, flags, features, ac, ac2, e1, e2):
    for j in flags:
        features.append(dic['features'][j])
        ac.append(dic['anisotropy_corr'][j])
        ac2.append(dic['anisotropy_corr_2'][j])
        e1.append(dic['e1_cal'][j])
        e2.append(dic['e2_cal'][j])
    return features, ac, ac2, e1, e2

def limit_on_data_merge(paths, dics, limit, name):
    mydir = config.workpath('Test5')
    SN = []
    flags = []
    features =[]
    ac = []
    ac2 =[]
    e1 = []
    e2 = []
    num_flags = 0
    for j,i in enumerate(dics):
        flags = []
        flags, SN = Flags(paths[j], i, SN, flags, limit)
        num_flags = num_flags + len(flags)
        features, ac, ac2, e1, e2 = Flagging(i, flags, features, ac, ac2, e1, e2)
    dic = {}
    dic['features'] = np.array(features)
    dic['anisotropy_corr'] = np.array(ac)
    dic['e1_cal'] = np.array(e1)
    dic['anisotropy_corr_2'] = np.array(ac2)
    dic['e2_cal'] = np.array(e2)
    dic['n_rea'] = dics[0]['n_rea']
    dic['n_fea'] = dics[0]['n_fea']
    dic['n_cas'] = num_flags
    dic['SN']= SN
    dic['c-bias_1']= cost_func_1(np.ones((num_flags,204000,1)),  dic['e1_cal'], dic['anisotropy_corr'])
    dic['c-bias_2']= cost_func_1(np.ones((num_flags,204000,1)),  dic['e2_cal'], dic['anisotropy_corr_2'])
    pickle.dump( dic, open( mydir + '/Processed_data_merged'+ name +'.p', "wb" ) )
    print(num_flags, dic['c-bias_1'],dic['c-bias_2'])
    return dic
       
    


# dic = Data_processing('Test2', 1, True)
def cost_func(preds,  polarisation, anisotropy_corr, polarisation2,  anisotropy_corr2):
    preds = tf.convert_to_tensor(preds, dtype=float)
    preds1,preds2 = tf.split(preds, num_or_size_splits=2, axis = 2)
    polarisation = tf.convert_to_tensor(polarisation, dtype=float)
    anisotropy_corr = tf.convert_to_tensor(anisotropy_corr, dtype=float)
    polarisation2 = tf.convert_to_tensor(polarisation2, dtype=float)
    anisotropy_corr2 = tf.convert_to_tensor(anisotropy_corr2, dtype=float)
    if tf.keras.backend.ndim(preds) == 3:  
        # print(Minus_sq)
        # print(tf.keras.backend.mean(Minus_sq, axis = 1, keepdims = True))
        cs = tf.keras.backend.mean(polarisation - preds1 * anisotropy_corr, axis = 1, keepdims = True)
        cs_square = tf.keras.backend.square(cs)
        cs2 = tf.keras.backend.mean(polarisation2 - preds2 * anisotropy_corr2, axis = 1, keepdims = True)
        cs_square2 = tf.keras.backend.square(cs2)
        cs_square_mean = tf.keras.backend.mean(cs_square, keepdims = True)
        cs_square_mean2 = tf.keras.backend.mean(cs_square2, keepdims = True) 
        return (cs_square_mean + cs_square_mean2)*10**6
    else:
        print('false number of dimensions')
        
def cost_func_1(preds,  polarisation, anisotropy_corr):
    preds = tf.convert_to_tensor(preds, dtype=float)
    polarisation = tf.convert_to_tensor(polarisation, dtype=float)
    anisotropy_corr = tf.convert_to_tensor(anisotropy_corr, dtype=float)
    if tf.keras.backend.ndim(preds) == 3:  
        cs = tf.keras.backend.mean(polarisation - preds * anisotropy_corr, axis = 1, keepdims = True)
        cs_square = tf.keras.backend.square(cs)
        cs_square_mean = tf.keras.backend.mean(cs_square, keepdims = False)
        return cs_square_mean*10**6
    else:
        print('false number of dimensions')

def create_model_1(nreas, nfea, hidden_layers = (3,3)):
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
    model.add_loss(cost_func_1(x, auxilary_fea1, auxilary_fea2))
    return model


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
checkpoint_path2 = config.workpath("training_2/cp.ckpt")
checkpoint_path3 = config.workpath("training_3/cp.ckpt")
checkpoint_path4 = config.workpath("training_4/cp.ckpt")
checkpoint_path5 = config.workpath("training_5/cp.ckpt")
checkpoint_path6 = config.workpath("training_6/cp.ckpt")
checkpoint_path7 = config.workpath("training_7/cp.ckpt")
checkpoint_path8 = config.workpath("training_8/cp.ckpt")
checkpoint_path9 = config.workpath("training_9/cp.ckpt")
checkpoint_path10 = config.workpath("training_10/cp.ckpt")
checkpoint_path11 = config.workpath("training_11/cp.ckpt")
checkpoint_path12 = config.workpath("training_12/cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_dir2 = os.path.dirname(checkpoint_path2)
checkpoint_dir3 = os.path.dirname(checkpoint_path3)
checkpoint_dir3 = os.path.dirname(checkpoint_path4)

def train(dic, checkpoint_path, epochs, hidden_layers = (3,3), learning_rate = 0.05):
    model = create_model(dic['n_rea'], dic['n_fea'], hidden_layers = (3,3))
    model.compile(
            loss = None,
            optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate),
            metrics = [])
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=0,
            monitor ='loss',
            save_freq='epoch',
            save_best_only=True)
    history = model.fit(
            [dic['features'], dic['e1_cal'], dic['anisotropy_corr'],dic['e2_cal'], dic['anisotropy_corr_2']], 
            None, 
            epochs = epochs,
            verbose = 0, 
            batch_size = None,
            callbacks=[cp_callback]) 
    model.load_weights(checkpoint_path)
    return model, history

def scheduler(epoch, lr):
    if epoch < 250:
        return lr
    elif epoch < 400:
        return 0.05
    else:
        return 0.01

def train_1(dic, checkpoint_path,checkpoint_path2, epochs, hidden_layers = (3,3)):
    model = create_model_1(dic['n_rea'], dic['n_fea'])
    model2 = create_model_1(dic['n_rea'], dic['n_fea'])
    model.compile(
            loss = None,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
            metrics = [])
    model2.compile(
            loss = None,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
            metrics = [])
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=0,
            monitor ='loss',
            save_freq='epoch',
            save_best_only=True)
    # LR_Scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    # Reduce_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, mode = 'min', verbose = 1,
    #                           patience=25, min_lr=0.01, cooldown = 0)
    cp_callback2 = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path2,
            save_weights_only=True,
            verbose=0,
            monitor ='loss',
            save_freq='epoch',
            save_best_only=True)
    # model.save_weights(checkpoint_path.format(epoch=0))
    # model2.save_weights(checkpoint_path2.format(epoch=0))
    history = model.fit(
            [dic['features'], dic['e1_cal'], dic['anisotropy_corr']], 
            None, 
            epochs = epochs,
            verbose = 0, 
            batch_size = None,
            callbacks=[cp_callback]) 
    history2 = model2.fit(
            [dic['features'], dic['e2_cal'], dic['anisotropy_corr_2']], 
            None, 
            epochs = epochs,
            verbose = 0, 
            batch_size = None,
            callbacks=[cp_callback2]) 
    model.load_weights(checkpoint_path)
    model2.load_weights(checkpoint_path2)
    #print(model.evaluate([dic['features'], dic['e1_cal'], dic['anisotropy_corr']]), model2.evaluate([dic['features'], dic['e2_cal'], dic['anisotropy_corr_2']]))
    return model, model2, history, history2

def validate(dic, checkpoint_path):
    # checkpoint_path = config.workpath("training_1/cp.ckpt")
    model = create_model(dic['n_rea'], dic['n_fea'])
    model.compile(
            loss = None,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
            metrics = [])
    model.load_weights(checkpoint_path)
    loss = model.evaluate([dic['features'], dic['e1_cal'], dic['anisotropy_corr'], dic['e2_cal'], dic['anisotropy_corr_2']])
    val_preds = model.predict(x = [dic['features'], dic['e1_cal'], dic['anisotropy_corr'], dic['e2_cal'], dic['anisotropy_corr_2']])
    return val_preds

def validate_1(dic, checkpoint_path, checkpoint_path2):
    # checkpoint_path = config.workpath("training_1/cp.ckpt")
    # checkpoint_path2 = config.workpath("training_2/cp.ckpt")
    model = create_model_1(dic['n_rea'], dic['n_fea'])
    model.compile(
            loss = None,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
            metrics = [])
    model.load_weights(checkpoint_path)
    model2 = create_model_1(dic['n_rea'], dic['n_fea'])
    model2.compile(
            loss = None,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
            metrics = [])
    model2.load_weights(checkpoint_path2)
    loss = model.evaluate([dic['features'], dic['e1_cal'], dic['anisotropy_corr']])
    val_preds = model.predict(x = [dic['features'], dic['e1_cal'], dic['anisotropy_corr']])
    loss2 = model2.evaluate([dic['features'], dic['e2_cal'], dic['anisotropy_corr_2']])
    val_preds2 = model2.predict(x = [dic['features'], dic['e2_cal'], dic['anisotropy_corr_2']])
    return val_preds, val_preds2


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
        

def cases_scatter_plot(dic,bsm, param, label,xlabel, name, first_component = True, non_boost=True):
    if first_component == True:
        e = (dic['e1_cal'] -  bsm * dic['anisotropy_corr'])
        e2 = (dic['e1_cal'] -  dic['anisotropy_corr'])
    else:
        e = (dic['e2_cal'] -  bsm * dic['anisotropy_corr_2'])
        e2 = (dic['e2_cal'] -  dic['anisotropy_corr_2'])        
    #params = oneD_to_threeD(path, 'Signal_to_Noise', dic['n_cas'], dic['n_rea'])
    # params = dic['SN']
    if param == 'aperture_sum':
        m = 0
        fea_max = 2000
    elif param == 'sigma_mom':
        m = 1
        fea_max = 40
    elif param == 'rho4_mom':
        m = 3
        fea_max = 2.6
    # params2 = dic_train['features'][:,:,0]
    params = dic['features'][:,:,m]*fea_max
    if param == 'sigma_mom':
        params = params*0.02
    error = []
    for j in range(dic['n_cas']):
        means = []
        mean = np.mean(e[j,:,0])
        for i in range(int(dic['n_rea']/120)):
            means.append(np.mean(e[j,i*120:(i+1)*120,0]))
        stabw = np.sum((means - mean)**2)/int(dic['n_rea']/120)
        stabw = np.sqrt(stabw)/np.sqrt(int(dic['n_rea']/120))
        error.append(stabw)   
    error2 = []
    for j in range(dic['n_cas']):
        means2 = []
        mean2 = np.mean(e2[j,:,0])
        for i in range(int(dic['n_rea']/120)):
            means2.append(np.mean(e2[j,i*120:(i+1)*120,0]))
        stabw = np.sum((means2 - mean2)**2)/int(dic['n_rea']/120)
        stabw = np.sqrt(stabw)/np.sqrt(int(dic['n_rea']/120))
        error2.append(stabw)       
    dic1 = mean_of(dic['n_cas'], e, e2, params)
    if param == 'aperture_sum':
        t_exp = 3*565 #s
        gain = 3.1 #e/ADU
        Z_p = 24.6
        dic1['param_mean'] = -1/0.4 * np.log10(gain/t_exp*dic1['param_mean'])+ Z_p
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # x = np.linspace(0.05,0.9,3)
    # ax.fill_between(x, -5.2*10**(-5), 5.2*10**(-5), color = 'darkgrey', label = 'Euclid requirement')
    if non_boost == False:
        plt.scatter(dic1['param_mean'], dic1['e2_mean'], marker = '.', color = 'r', label = '$b^{sm} = 1$')
        r1 = np.mean(np.square(dic1['e2_mean']))
        #print(r1)
        plt.errorbar(dic1['param_mean'], dic1['e2_mean'], yerr = error2, ecolor = 'r', linestyle = ' ')
        plt.xlabel(xlabel)
        plt.ylabel('$c_1$ additive bias')
        plt.legend(loc = 'best')
        plt.axhline(y = 0, color = 'black', linestyle = 'dashed')
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.tight_layout()
        # plt.savefig('Thesis_plots/Scatter_Colloq2.png')
        # plt.ylim([-0.0025, 0.0025])
        # plt.savefig('Plots2/C-bias2/training_example0.png')
    # plt.scatter(dic1['param_mean'], dic1['e2_mean'], label = '$b^{sm} = 1$')
    plt.scatter(dic1['param_mean'], dic1['e_mean'], s = 40, marker = '.',color = 'r', label = label)
    r2 = np.mean(np.square(dic1['e_mean']))
    #print(r2)        plt.savefig('Thesis_plots/Scatter_Colloq2.png')
    # print(r2/r1)
    print(param)
    plt.errorbar(dic1['param_mean'], dic1['e_mean'], yerr = error, ecolor = 'r', linestyle = ' ')
    param_min = min(dic1['param_mean'])
    param_max= max(dic1['param_mean'])
    plt.axhline(y = 0, color = 'black', linestyle = 'dashed')
    #ax.set_yscale("log")
    # plt.xlabel('$e_1^{PSF}$ psf polarisation')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.xlabel(xlabel)
    # plt.xlabel('Standard deviation of additive bias estimate')
    matplotlib.scale.SymmetricalLogScale(ax)
    if first_component == True:
        plt.ylabel('$c_1$ additive bias')
    else:
        plt.ylabel('$c_2$ additive bias')
    # plt.axis('scaled')
    # plt.tight_layout()
    plt.legend(loc = 'best')
    plt.tight_layout()
    # plt.savefig('Thesis_plots/Scatter_Colloq_train_limit_log.png')
    plt.savefig('Thesis_plots/Scatter_case_Euclid_reqLog' + name + '.pdf')
    plt.clf()
    return None


def cases_scatter_plot_highM(dic,bsm,bsm2, path, param, xlabel, name, first_component = True, non_boost=True):
    if first_component == True:
        e2 = (dic['e1_cal'] -  bsm * dic['anisotropy_corr'])
        e = (dic['e1_cal'] -  dic['anisotropy_corr'])
        e3 = (dic['e1_cal'] -  bsm2 * dic['anisotropy_corr'])
    else:
        e2 = (dic['e2_cal'] -  bsm * dic['anisotropy_corr_2'])
        e = (dic['e2_cal'] -  dic['anisotropy_corr_2'])        
        e3 = (dic['e2_cal'] -  bsm2 * dic['anisotropy_corr_2'])
    # params = oneD_to_threeD(path, param, dic['n_cas'], dic['n_rea'])
    if param == 'aperture_sum':
        m = 0
    elif param == 'sigma_mom':
        m = 1
    elif param == 'rho4_mom':
        m = 3
    if param == 'aperture_sum':
        m = 0
        fea_max = 2000
    elif param == 'sigma_mom':
        m = 1
        fea_max = 40
    elif param == 'rho4_mom':
        m = 3
        fea_max = 2.6
    params = dic['features'][:,:,m]*fea_max
    # print(params)
    if param == 'sigma_mom':
        params = params*0.02
    if param == 'aperture_sum':
        t_exp = 3*565 #s
        gain = 3.1 #e/ADU
        Z_p = 24.6
        params = -1/0.4 * np.log10(gain/t_exp*params)+ Z_p
    error = []
    for j in range(dic['n_cas']):
        means = []
        mean = np.mean(e[j,:,0])
        for i in range(int(dic['n_rea']/120)):
            means.append(np.mean(e[j,i*120:(i+1)*120,0]))
        stabw = np.sum((means - mean)**2)/int(dic['n_rea']/120)
        stabw = np.sqrt(stabw)/np.sqrt(int(dic['n_rea']/120))
        error.append(stabw)   
    error2 = []
    for j in range(dic['n_cas']):
        means2 = []
        mean2 = np.mean(e2[j,:,0])
        for i in range(int(dic['n_rea']/120)):
            means2.append(np.mean(e2[j,i*120:(i+1)*120,0]))
        stabw = np.sum((means2 - mean2)**2)/int(dic['n_rea']/120)
        stabw = np.sqrt(stabw)/np.sqrt(int(dic['n_rea']/120))
        error2.append(stabw)       
    error3 = []
    for j in range(dic['n_cas']):
        means2 = []
        mean2 = np.mean(e3[j,:,0])
        for i in range(int(dic['n_rea']/120)):
            means2.append(np.mean(e3[j,i*120:(i+1)*120,0]))
        stabw = np.sum((means2 - mean2)**2)/int(dic['n_rea']/120)
        stabw = np.sqrt(stabw)/np.sqrt(int(dic['n_rea']/120))
        error3.append(stabw)   
    dic1 = mean_of(dic['n_cas'], e, e2, params)
    dic2 = mean_of(dic['n_cas'], e, e3, params)
    if non_boost == True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(dic1['param_mean'], dic1['e_mean'], s= 12, marker = '.', color = 'r', label = '$b^{sm} = 1$')
        r1 = np.mean(np.square(dic1['e_mean']))
        #print(r1)
        plt.errorbar(dic1['param_mean'], dic1['e_mean'], yerr = error2, ecolor = 'r', linestyle = ' ', elinewidth = 1)
        plt.xlabel('$e_1^{PSF}$ psf polarisation')
        plt.ylabel('$c_1$ additive bias')
        plt.legend(loc = 'upper center')
        # plt.ylim([-0.0025, 0.0025])
        # plt.savefig('Plots2/C-bias2/training_example0.png')
    # plt.scatter(dic1['param_mean'], dic1['e2_mean'], label = '$b^{sm} = 1$')
    plt.scatter(dic1['param_mean'], dic1['e2_mean'],s=12, marker = '.',color = 'orange', label = 'ML without $Q^4_{PSF}$')
    plt.scatter(dic1['param_mean'], dic2['e2_mean'],s=12,  marker = '.',color = 'green', label = 'ML with $Q^4_{PSF}$')
    r2 = np.mean(np.square(dic1['e2_mean']))
    r3 = np.mean(np.square(dic2['e2_mean']))
    #print(r2)
    print(r1)
    print(r2/r1, r3/r1, r3/r2)
    print(param)
    plt.errorbar(dic1['param_mean'], dic1['e2_mean'], yerr = error, ecolor = 'orange', linestyle = ' ', elinewidth = 1)
    plt.errorbar(dic2['param_mean'], dic2['e2_mean'], yerr = error3, ecolor = 'green', linestyle = ' ', elinewidth = 1)
    plt.axhline(y = 0, color = 'black', linestyle = 'dashed')
    # plt.xlabel('$e_1^{PSF}$ psf polarisation')
    plt.xlim([0.12, 0.32])
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.xlabel(xlabel)
    # plt.xlabel('Standard deviation of additive bias estimate')
    if first_component == True:
        plt.ylabel('$c_1$ additive bias')
    else:
        plt.ylabel('$c_2$ additive bias')
    # plt.axis('scaled')
    # plt.tight_layout()
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig('Thesis_plots/Comparison_M_noM_zoom' + name + '.pdf')
    # plt.savefig('Thesis_plots/Comparison_M_noM_' + name + '.pdf')
    plt.clf()
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
# # print(dic6)
# dic_5 = Data_processing('Test5_validation')
path = config.workpath('Test5')
path2 = config.workpath('Test5_validation')
path3 = config.workpath('Test3')
path4 = config.workpath('Test2')
path5 = config.workpath('Test')
path6 = config.workpath('Test4')
# dic_val2 = Data_processing(config.workpath('Test2'))
#dic_train2 = Data_processing(path2, True, False)
#dic = Data_processing(path3)
#dic2 = Data_processing(path4)
#dic = Data_processing(path5)
# dic4 = Data_processing(path6)
# dic3 = Data_processing(path3)
#dic4 = pickle.load( open(path6 + "/Processed_data_withM.p", "rb" ) )
# # # dic4 = pickle.load( open(path6 + "/Processed_data_noM.p", "rb" ) )
# dic3 = pickle.load( open(path3 + "/Processed_data_withM.p", "rb" ) )
# # # # dic5 = pickle.load( open(path + "/Processed_data_noM.p", "rb" ) )
# dic5 = pickle.load( open(path + "/Processed_data_withM.p", "rb" ) )
# #dic4 = pickle.load( open(path6 + "/Processed_data_withM.p", "rb" ) )
# # # #print('start')
# # # #dic3 = Data_processing(path3)
# # # #dic3 = pickle.load( open(path3 + "/Processed_data_withM.p", "rb" ) )
# dic5_val = pickle.load( open(path2 + "/Processed_data_withM.p", "rb" ) )
#dic_train2 = pickle.load( open(path2 + "/Processed_data_withM.p", "rb" ) )
#print(cost_func_1(np.ones((dic_train2['n_cas'],204000,1)), dic_train2['e1_cal'], dic_train2['anisotropy_corr']))
#print(cost_func_1(np.ones((dic_train2['n_cas'],204000,1)), dic_train2['e2_cal'], dic_train2['anisotropy_corr_2']))
#dic_merge = pickle.load( open(path + "/Processed_data_mergedtest.p", "rb" ) )
# dic_merge = limit_on_data_merge([path6, path, path2], [dic4,dic5,dic5_val],10, 'NNcheck_SN10_moretrain_train')
# dic_merge_val = limit_on_data_merge([path3], [dic3],10, 'NNcheck_SN10_moretrain_val')
# dic_merge = pickle.load( open(path + "/Processed_data_mergedNNcheck_SN0_noM_train.p", "rb" ) )
# dic_merge_val = pickle.load( open(path + "/Processed_data_mergedNNcheck_SN0_noM_val.p", "rb" ) )

# dic = pickle.load( open(path + "/Processed_data_withM.p", "rb" ) )
# bsms = Table.read(path + '/BSM_1203_train.fits')
# bsms2 = Table.read(path + '/BSM_Ms_1403_train.fits')
# cases_scatter_plot_highM(dic, bsms['bsm1'], bsms2['bsm1'], path, 'sigma_mom', 'Galaxy size in arcsec', 'train1_sigma')
#print(dic_merge['c-bias_1'] + dic_merge['c-bias_2'])
#print(dic_merge_val['c-bias_1'] + dic_merge_val['c-bias_2'])
dic_merge = pickle.load( open(path + "/Processed_data_mergedNNcheck_SN0_train.p", "rb" ) )
dic_merge_val = pickle.load( open(path + "/Processed_data_mergedNNcheck_SN0_val.p", "rb" ) )
# bsm1_val, bsm2_val = validate_1(dic_merge_val, checkpoint_path8, checkpoint_path9)
# bsm1, bsm2 = validate_1(dic_merge, checkpoint_path8, checkpoint_path9)
# bsm1, bsm2 = np.split(table,2)
#print(dic_merge['c-bias_1'] + dic_merge['c-bias_2'])
#print(dic_merge_val['c-bias_1'] + dic_merge_val['c-bias_2'])
#dic_merge = pickle.load( open(path + "/Processed_data_mergedNNcheck_SN10_moretrain_train.p", "rb" ) )
#dic_merge_val = pickle.load( open(path + "/Processed_data_mergedNNcheck_SN10_moretrain_val.p", "rb" ) )
#print(dic_merge['c-bias_1'] + dic_merge['c-bias_2'])
#print(dic_merge_val['c-bias_1'] + dic_merge_val['c-bias_2'])
# validate_1(dic5, checkpoint_path2, checkpoint_path3)

# dic_val = pickle.load( open(path2 + "/Processed_valdata_withM.p", "rb" ) )
#dic1_SN = limit_on_data_merge([dic], 10, '1')
#dic2_SN = limit_on_data_merge([dic2], 10, '2')
# dic_merge = limit_on_data_merge([dic3,dic5, dic5_val, dic4], 0, 'plot_merger')
#dic4_SN = limit_on_data_merge([dic4], 10, '4')
#dic5_SN = limit_on_data_merge([dic5], 10, '5')
# dic_merge_val = limit_on_data_merge([dic5_val, dic4], 0, 'val_nolimit')
# print(cost_func_1(np.ones((dic_val['n_cas'],204000,1)), dic_val['e1_cal'], dic_val['anisotropy_corr']))
# print(cost_func_1(np.ones((dic_val['n_cas'],204000,1)), dic_val['e2_cal'], dic_val['anisotropy_corr_2']))
#dic_merge = pickle.load( open(config.workpath('Test5/') + "Processed_data_merged'+ name +'.p", "rb" ) )
print('done start SN0 noM')
# print('data loaded')
# dic_train = limit_on_flux(dic_train2)
# dic_val = limit_on_flux(dic_val)
# print('data filtered')
# print(dic_train['e1_cal'][0])
# print(dic_train2['features'][0])
#print(dic_train['n_rea'], dic_train['n_fea'], dic_train['n_cas'])
# model, model2, history, history2 = train_1(dic_merge,checkpoint_path7, checkpoint_path8, 1000)
# bsm = model2.predict(x = [dic_merge['features'], dic_merge['e1_cal'], dic_merge['anisotropy_corr']])
# model2.evaluate([dic_merge['features'], dic_merge['e2_cal'], dic_merge['anisotropy_corr_2']])
# model.evaluate([dic_merge['features'], dic_merge['e1_cal'], dic_merge['anisotropy_corr']]) 
# bsm1, bsm2 = validate_1(dic_merge_val, checkpoint_path7, checkpoint_path8)
#bsm, bsm2 = validate_1(dic_merge, checkpoint_path7, checkpoint_path8)
# # tab = Table.read(config.workpath('Test5') + '/BSM_Ms_1403_2train.fits')
# cases_scatter_plot(dic_merge, bsm2, 'sigma_mom', 'ML boosted', 'Galaxy size in arcsec', 'c2', False)
# cases_scatter_plot(dic_merge, bsm1, 'sigma_mom', 'ML boosted', 'Galaxy size in arcsec', 'c1')
#dic_val = pickle.load( open(path2 + "/Processed_data_noM.p", "rb" ) )
#dic_train = pickle.load( open(path + "/Processed_data_noM.p", "rb" ) )
# #dic_train2 = pickle.load( open(path + "/Processed_data_new.p", "rb" ) )


def Bin_SN_error(dic, bin_num):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    e = (dic['e1_cal'] -  dic['anisotropy_corr'])
    SN = dic['SN']
    error = []
    for j in range(dic['n_cas']):
        means = []
        mean = np.mean(e[j,:,0])
        for i in range(int(dic['n_rea']/120)):
            means.append(np.mean(e[j,i*120:(i+1)*120,0]))
        stabw = np.sum((means - mean)**2)/int(dic['n_rea']/120)
        stabw = np.sqrt(stabw)/np.sqrt(int(dic['n_rea']/120))
        error.append(stabw)
    ls = np.linspace(0, 1,bin_num)
    qs = np.quantile(SN, ls)
    dig = np.digitize(SN, qs)
    bins = init_list_of_objects(bin_num)
    bins2 = init_list_of_objects(bin_num)
    for i in range(dic['n_cas']):
        bins[dig[i]-1].append(error[i])
        bins2[dig[i]-1].append(SN[i])
    mean1 = []
    mean2 = []
    errors = []
    for i in range(bin_num):
        mean1.append(np.mean(bins[i]))
        errors.append(np.sqrt(np.sum((bins[i] - np.mean(bins[i]))**2)/bin_num))
        mean2.append(np.mean(bins2[i]))
    plt.scatter(mean2, mean1)
    # plt.errorbar(mean2, mean1, yerr = errors, linestyle = ' ')
    plt.xlabel('Signal to Noise ratio')
    plt.ylabel('Statistical error')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.savefig('Thesis_plots/SN_error.pdf')
    plt.show()

        
# Bin_SN_error(dic_merge, 8)  
    

# model, history = train(dic_train, checkpoint_path, 2000)
# bsm = model.predict(x = [dic_train['features'], dic_train['e1_cal'], dic_train['anisotropy_corr'], dic_train['e2_cal'], dic_train['anisotropy_corr_2']])
# bsm_val = validate(dic_val, checkpoint_path)
# tab = Table([bsm], names = ['train'], dtype = ['f4'])
# tab2 = Table([bsm_val], names = ['val'], dtype = ['f4'])
# file_name = os.path.join(path, 'BSM_tog_train2000.fits')
# file_name = os.path.join(path, 'BSM_1203_train.fits')
# # tab.write( file_name , overwrite=True)  
# # file_name2 = os.path.join(path, 'BSM_tog_val2000.fits')
# file_name2 = os.path.join(path, 'BSM_1203_val.fits')
# # tab2.write( file_name , overwrite=True)  
# tab = Table.read(file_name)
# # bsm = tab['train']
# bsm1 = tab['bsm1']
# bsm2 = tab['bsm2']
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # # bsm1, bsm2 = np.split(bsm,2)
# plt.scatter(np.mean(bsm1, axis =1), np.mean(bsm2, axis =1), marker = '.')
# a = np.linspace(0, 1.2, 5)
# plt.plot(a, analyze.linear(a, 1, 0), color = 'r')
# plt.xlabel('$b^{sm}_1$')
# plt.ylabel('$b^{sm}_2$')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.tight_layout()
# plt.savefig('Thesis_plots/BSMs.pdf')
# plt.clf()
# # tab = Table.read(file_name2)
# # # bsm_val = tab['val']
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # bsm1 = tab['bsm_val1']
# # bsm2 = tab['bsm_val2']
# # # bsm1, bsm2 = np.split(bsm_val,2)
# plt.scatter(np.mean(bsm1_val, axis =1), np.mean(bsm2_val, axis =1), marker = '.')
# plt.plot(a, analyze.linear(a, 1, 0), color = 'r')
# plt.xlabel('$b^{sm}_1$')
# plt.ylabel('$b^{sm}_2$')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.tight_layout()
# plt.savefig('Thesis_plots/BSMs_val.pdf')
# plt.clf()
def check_MLvars(dic, dic_val, checkpoint_path, number, hidden_layers = (3,3), learning_rate = 0.05):
    # t = Table(names = ['train', 'val'], dtype = ['f4', 'f4'])
    t = Table.read(config.workpath('Test5') + '/No_Split_SN0_0604.fits')
    results = []
    results_val = []
    for i in range(number):
        model,history = train(dic, checkpoint_path, 2000, hidden_layers = hidden_layers, learning_rate = learning_rate)
        bsm = model.predict(x = [dic['features'], dic['e1_cal'], dic['anisotropy_corr'], dic['e2_cal'], dic['anisotropy_corr_2']])
        bsm_val = validate(dic_val, checkpoint_path)
        results = cost_func(bsm,  dic['e1_cal'], dic['anisotropy_corr'], dic['e2_cal'], dic['anisotropy_corr_2'])
        results_val = cost_func(bsm_val,  dic_val['e1_cal'], dic_val['anisotropy_corr'], dic_val['e2_cal'], dic_val['anisotropy_corr_2'])
        t.add_row([results, results_val])
        t.write(config.workpath('Test5/No_Split_SN0_0904.fits'), overwrite = True)
        print(cost_func(bsm,  dic['e1_cal'], dic['anisotropy_corr'], dic['e2_cal'], dic['anisotropy_corr_2']), cost_func(bsm_val,  dic_val['e1_cal'], dic_val['anisotropy_corr'], dic_val['e2_cal'], dic_val['anisotropy_corr_2']))
    print('finished with ' + str(np.mean(results)) + 'and ' + str(np.mean(results_val)))
    return None

def check_MLvars_split(dic, dic_val, checkpoint_path, checkpoint_path2, number, hidden_layers = (3,3)):
    #table = Table(names = ['train', 'val'], dtype = ['f4', 'f4'])
    table = Table.read(config.workpath('Test5') + '/SN10_0404.fits')
    results = []
    results_val = []
    for i in range(number):
        model, model2, history, history2 = train_1(dic, checkpoint_path, checkpoint_path2, 1000, hidden_layers = hidden_layers)
        bsm = model.predict(x = [dic['features'], dic['e1_cal'], dic['anisotropy_corr']])
        bsm2 = model2.predict(x = [dic['features'], dic['e2_cal'], dic['anisotropy_corr_2']])
        bsm1_val, bsm2_val = validate_1(dic_val, checkpoint_path, checkpoint_path2)
        results = cost_func_1(bsm,  dic['e1_cal'], dic['anisotropy_corr']) + cost_func_1(bsm2,  dic['e2_cal'], dic['anisotropy_corr_2'])
        results_val = cost_func_1(bsm1_val,  dic_val['e1_cal'], dic_val['anisotropy_corr']) + cost_func_1(bsm2_val,  dic_val['e2_cal'], dic_val['anisotropy_corr_2'])
        table.add_row([results, results_val])
        table.write(config.workpath('Test5/SN10_0604.fits'), overwrite = True)
        print(cost_func_1(bsm,  dic['e1_cal'], dic['anisotropy_corr']), cost_func_1(bsm2,  dic['e2_cal'], dic['anisotropy_corr_2']), cost_func_1(bsm1_val,  dic_val['e1_cal'], dic_val['anisotropy_corr']), cost_func_1(bsm2_val,  dic_val['e2_cal'], dic_val['anisotropy_corr_2']))
    print('finished')
    return None

def check_MLvars_NN_lrs(dic, epochs, tab, lrs, number, hidden_layers = (3,3)):
    res = []
    checkpoint_path = config.workpath("training_lrs/cp_" + str(epochs)  + '_'+ str(number)  + ".ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print('start number ' + str(number)+ ' in epoch ' + str(epochs))
    for i in lrs:
        model, history = train(dic, checkpoint_path, epochs, hidden_layers = hidden_layers, learning_rate=i)
        bsm = model.predict(x = [dic['features'], dic['e1_cal'], dic['anisotropy_corr']])
        results = cost_func_1(bsm,  dic['e1_cal'], dic['anisotropy_corr'])
        res.append(results)
        print(results)
    tab.add_row([res[0], res[1], res[2], res[3], res[4], res[5]])
    print('done with number ' + str(number) + ' in epoch ' + str(epochs))
    return None

# def check_MLvars_NN_layers(dic, checkpoint_path, epochs, tab, layers, number, hidden_layers = (3,3)):
#     import tensorflow as tf
#     res = []
#     # checkpoint_path = config.workpath("training_layers/cp_" + str(epochs)  + '_'+ str(number)  + ".ckpt")
#     # checkpoint_dir = os.path.dirname(checkpoint_path)
#     print('start number ' + str(number)+ ' in epoch ' + str(epochs))
#     for i in layers:
#         model, history = train(dic, checkpoint_path, epochs, hidden_layers = (i,i), learning_rate=0.05)
#         bsm = model.predict(x = [dic['features'], dic['e1_cal'], dic['anisotropy_corr']])
#         results = cost_func_1(bsm,  dic['e1_cal'], dic['anisotropy_corr'])
#         res.append(results)
#         print(results)
#     tab.add_row([res[0], res[1], res[2], res[3], res[4], res[5]])
#     print('done with number ' + str(number) + ' in epoch ' + str(epochs))
#     return None

# def ML_check_lrs(epochs):
#     lrs =[0.2,0.1,0.05,0.02,0.01,0.005]
#     final = []
#     tab = Table(names=['0.2','0.1','0.05','0.02','0.01','0.005'], dtype = ['f4', 'f4', 'f4', 'f4', 'f4', 'f4'])
#     N =50
#     for i in range(N):
#         check_MLvars_NN_lrs(dic_train, epochs, tab, lrs, i)
#     tab.write(config.workpath('Test5/lrs_' + str(epochs) + '.fits'), overwrite = True)

# def ML_check_layers(epochs):
#     tab2 = Table.read(config.workpath('Test5/layers_' + str(epochs) + '.fits'))
#     layers = [1,2,3,4,5,6]
#     final = []
#     tab = Table(names=['1','2','3','4','5','6'], dtype = ['f4', 'f4', 'f4', 'f4', 'f4', 'f4'])
#     N =25
#     for i in range(N):
#         check_MLvars_NN_layers(dic_train, checkpoint_path2, epochs, tab, layers, i)
#         tab3 = vstack(tab2, tab)
#         tab3.write(config.workpath('Test5/layers_added_' + str(epochs) + '.fits'), overwrite = True)
    
# def ML_check_layers_pool(epochs):
#     layers = [1,2,3,4,5,6]
#     final = []
#     tab = Table(names=['1','2','3','4','5','6'], dtype = ['f4', 'f4', 'f4', 'f4', 'f4', 'f4'])
#     N =40
#     for i in range(N):
#         final.append([dic_train, epochs, tab, layers, i])
#     with Pool(processes=5) as pool:
#         pool.starmap(check_MLvars_NN_layers, final)
#     tab.write(config.workpath('Test5/layers_' + str(epochs) + '.fits'), overwrite = True)
    
# def ML_check_lrs_pool(epochs):
#     lrs =[0.2,0.1,0.05,0.02,0.01,0.005]
#     final = []
#     tab = Table(names=['1','2','3','4','5','6'], dtype = ['f4', 'f4', 'f4', 'f4', 'f4', 'f4'])
#     N =40
#     for i in range(N):
#         final.append([dic_train, epochs, tab, lrs, i])
#     with Pool(processes=5) as pool:
#         pool.starmap(check_MLvars_NN_lrs, final)
#     tab.write(config.workpath('Test5/lrs_' + str(epochs) + '.fits'), overwrite = True)


#check_MLvars_split(dic_merge, dic_merge_val, checkpoint_path4, checkpoint_path5, 20)
check_MLvars(dic_merge, dic_merge_val, checkpoint_path3, 20)


# ML_check_lrs_pool(1000)
#ML_check_lrs(1000)
# ML_check_layers(500)
# ML_check_layers(1000)
# ML_check_layers(1500)

# for i in lrs:
#     res.append(check_MLvars_NN(dic_train, checkpoint_path3, 25, 500, hidden_layers = (3,3), learning_rate=i))
# t = Table([res], names = ['500'], dtype = ['f4'])
# t.write(config.workpath('Test5/lrs_500.fits'), overwrite = True)
# for i in lrs:
#     res1000.append(check_MLvars_NN(dic_train, checkpoint_path3, 25, 1000, hidden_layers = (3,3), learning_rate=i))
# t = Table([res1000], names = ['1000'], dtype = ['f4'])
# t.write(config.workpath('Test5/lrs_1000.fits'), overwrite = True)
# for i in lrs:
#     res1500.append(check_MLvars_NN(dic_train, checkpoint_path3, 25, 1500, hidden_layers = (3,3), learning_rate=i))
# t = Table([res], names = ['1500'], dtype = ['f4'])
# t.write(config.workpath('Test5/lrs_1500.fits'), overwrite = True)
# t = Table([res1], names = ['tog'], dtype = ['f4'])
# t.write(config.workpath('Test5/tog.fits'), overwrite = True)
#tab = Table.read(config.workpath('Test5/split.fits'))
#print(tab)
#print(np.mean(np.sum(tab['split'], axis = 1)))

# t = Table(names = ['No_Split_train', 'No_Split_val'], dtype = ['f4', 'f4'])
# check_MLvars(dic_train, dic_val, checkpoint_path4, 100, t)
# t = Table( names = ['No_Split_train', 'No_Split_val'], dtype = ['f4', 'f4'])
# t.write(config.workpath('Test5/No_Split_2003.fits'), overwrite = True)

# t2 = Table(names = ['Ms_train', 'Ms_val'], dtype = ['f4', 'f4'])
# check_MLvars_split(dic_train, dic_val, checkpoint_path5, checkpoint_path6, 100, t2)
# t2 = Table([res2], names = ['No_Ms'], dtype = ['f4'])
# t2.write(config.workpath('Test5/No_Ms_2003.fits'), overwrite = True)

# tab = Table.read(path + '/BSM_train.fits')
# cases_scatter_plot(dic_train, tab['bsm1'], path, 'M4_1_psf_adap', 'ML boosted', 'adapted fourth moments')     
# dics = [dic_train, dic_train2]
# result = []
# for i in dics:
#     result.append(check_MLvars(i, checkpoint_path3, 10))

# table = Table([result], names = ['loss_M'], dtype = ['f4'])
# file_name = config.workpath('Test5/MLcheck_M4.fits')
# table.write(file_name, overwrite = True)

# lrs =[0.2,0.1,0.05,0.02,0.01,0.005]
# lrs2 =['0.2','0.1','0.05','0.02','0.01','0.005']
# ls = ['1', '2', '3', '4', '5', '6']
# layers = [1,2,3,4,5,6]
# result2 = []
# plt.rcParams.update({'font.size': 30})
# tab = Table.read(path + '/layers_500.fits')
# tab = Table.read(path + '/MLcheck_lrs.fits')
# tab2 = Table.read(path + '/MLcheck_lrs_1000.fits')
# # print(np.array(tab))
# # tab = Table.read(path + '/Ms_0603.fits')
# # tab2 = Table.read(path + '/layers_1000.fits')
# means = []
# means2 = []
# std = []
# std2 = []
# for i in range(6):
#     mean = np.mean(tab['loss_lr'][i])
#     means.append(mean)
#     std.append(np.sqrt(np.sum((tab['loss_lr'][i]-mean)**2)/(24*25)))
#     mean2 = np.mean(tab2['loss_lr'][i])10
#     means2.append(mean2)
#     std2.append(np.sqrt(np.sum((tab2['loss_lr'][i]-mean)**2)/(24*25)))
# print(means, std)
# print(means2,# print(dic_train2['features'][2]) std2)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(lrs, means, marker = '.', label = '$epochs = 500$')
# plt.scatter(lrs, means2, marker = '.', label = '$epochs = 1000$')
# plt.errorbar(lrs, means, yerr = std)
# plt.errorbar(lrs, means2, yerr = std2)
# # plt.scatter(tab['M4_1_psf_adap'], tab['M4_1_psf'], marker = '.')
# plt.xlabel('learning rate')
# plt.ylabel('average loss in $10^{-6}$')
# plt.legend()
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.show()

def mean(param, ncas = 10):
    print('start')
    means = []
    for i in range(ncas):
        tab = Table.read(config.workpath('M4_Check') + '/Measured_ksb_'+ str(i) + '.fits')
        means.append(tab[param])
    print('done')
    return means
    


# tab = Table.read(config.workpath('M4_Check') + '/Measured_ksb.fits')
# plt.scatter(mean('e2_cal_psf_adap'), mean('e2_cal_psf'))
# plt.xlabel('psf polarisation with adapted weight function')
# plt.ylabel('psf polarisation with circular weight function')


# dic_train = limit_on_flux(dic_train, dic_val)


#for i in lrs:
#    result2.append(check_MLvars(dic_train, checkpoint_path3, 5, hidden_layers=(3,3), learning_rate = i))

#table = Table([result2], names = ['loss_lr'], dtype = ['f4'])
#file_name = config.workpath('Test5/MLcheck_lrs_1000.fits')
#table.write(file_name, overwrite = True)

#table = Table.read(config.workpath('Test5/MLcheck_M4.fits'))
# means = []
# std = []
# for i in range(2):
# 	mean = np.mean(table['loss_M'][i])
# 	means.append(mean)
# 	std.append(np.sqrt(np.sum((table['loss_M'][i]-mean)**2)/10))
# print(means)
# print(std)


# # print(dic5)
# # bsms = validate(dic5, checkpoint_path)
# # dic6 = pickle.load( open(path + "/Processed_data.p", "rb" ) )
# # # # bsm = np.ones((100,207000,1))


# checkpoint_path4 = config.workpath("training_4/cp.ckpt")
# checkpoint_path5 = config.workpath("training_5/cp.ckpt")
# model, model2, history, history2 = train_1(dic_train, checkpoint_path4, checkpoint_path5, 1000)
#bsm1 = model.predict(x = [dic_train['features'], dic_train['e1_cal'], dic_train['anisotropy_corr']])
# bsm2 = model2.predict(x = [dic_train['features'], dic_train['e2_cal'], dic_train['anisotropy_corr_2']])
# table = Table.read(path + '/BSM_Ms_1403_train.fits')
# table2 = Table.read(path + '/BSM_Ms_1403_val.fits')
# bsm_val1, bsm_val2 = validate_1(dic_val, checkpoint_path4, checkpoint_path5)
# path = config.workpath('Test5')
# file_name = os.path.join(path, 'BSM_Ms_1403_2train.fits')
# file_name2 = os.path.join(path, 'BSM_Ms_1403_2val.fits')
# t = Table([table['bsm1'], bsm2], names = ['bsm1', 'bsm2'], dtype =['f4', 'f4'])
# t2 = Table([table2['bsm_val1'], bsm_val2], names = ['bsm_val1', 'bsm_val2'], dtype =['f4', 'f4'])
# t.write(file_name, overwrite = True)
# t2.write(file_name2, overwrite = True)
# print('data acquired')
# table = Table.read(path + '/BSM_Ms_1403_2train.fits')
# # table2 = Table.read(path + '/BSM_Ms_1403_2val.fits')
# # table = Table.read(path + '/BSM_1203_train.fits')
# # table2 = Table.read(path + '/BSM_1203_val.fits')
# # # # print('bsm acquired')
# params = ['aperture_sum', 'sigma_mom']
# names = ['Magnitude', 'Galaxy size in arcsec']
# # for i in range(2):
# #     cases_scatter_plot(dic_train, table['bsm1'], path,params[i], 'ML boost-factor' , names[i], name = 'train1' + params[i])
# #     cases_scatter_plot(dic_train, table['bsm2'], path,params[i], 'ML boost-factor' , names[i], name = 'train2' + params[i], first_component= False)
# #     cases_scatter_plot(dic_val, table2['bsm_val1'], path2,params[i], 'ML boost-factor' , names[i], name = 'val1' + params[i])
# #     cases_scatter_plot(dic_val, table2['bsm_val2'], path2,params[i], 'ML boost-factor' , names[i], name = 'val2' + params[i], first_component = False)

# cases_scatter_plot(dic_train, table['bsm1'], path,params[0], 'ML boost-factor' , names[0], name = 'train1' + params[1])

# table = Table.read(path + '/BSM_Ms_1403_2train.fits')
# table2 = Table.read(path + '/BSM_Ms_1403_2val.fits')
# table3 = Table.read(path + '/BSM_1203_train.fits')
# table4 = Table.read(path + '/BSM_1203_val.fits')

# cases_scatter_plot_highM(dic_train, table3['bsm1'], table['bsm1'], path, 'sigma_mom', 'Galaxy size in arcsec', name = 'train1_sigma')
# cases_scatter_plot_highM(dic_train, table3['bsm2'],table['bsm2'], path,'sigma_mom', 'Galaxy size in arcsec', name = 'train2_sigma', first_component= False)
# cases_scatter_plot_highM(dic_val, table4['bsm_val2'], table2['bsm_val2'], path2,'sigma_mom', 'Galaxy size in arcsec', name = 'val2_sigma', first_component= False)
# cases_scatter_plot_highM(dic_val,table4['bsm_val1'], table2['bsm_val1'], path2,'sigma_mom', 'Galaxy size in arcsec', name = 'val1_sigma')



# params = oneD_to_threeD(path, 'anisotropy_corr', dic_train['n_cas'], dic_train['n_rea'])

# dic = mean_of(100, dic_train['e1_cal']-table['bsm1']*dic_train['anisotropy_corr'], dic_train['e1_cal']-dic_train['anisotropy_corr'], params)
# meanss = []
# means2 = []
# error = []
# for j in range(dic_train['n_cas']):
#     means = []
#     mean = np.mean(table['bsm1'][j,:,0])
#     for i in range(int(dic_train['n_rea']/120)):
#         means.append(np.mean(table['bsm1'][j,i*120:(i+1)*120,0]))
#     stabw = np.sum((means - mean)**2)/int(dic_train['n_rea']/120)
#     stabw = np.sqrt(stabw)/np.sqrt(int(dic_train['n_rea']/120))
#     error.append(stabw)   name
# for i in range(100):
#     meanss.append(np.mean(table['bsm1'][i][:][0]))
#     means2.append(np.mean(params[i][:][0]))
# plt.scatter(dic['e2_mean'], meanss, marker = '.')
# # plt.errorbar(means2, meanss, yerr = error, linestyle = ' ')
# # print(error)
# plt.xlabel('additive bias $c_1$')
# plt.ylabel('$b^{sm}_1$')
# plt.rcParams.update({'font.size': 35})
# t_exp = 3*565 #s
# gain = 3.1 #e/ADU
# Z_p = 24.6
# for i in range(100):
#     path2 = config.workpath(path + '/Measured_ksb_' + str(i) +'.fits')
#     table =Table.read(path2)
#     #print(table['r_half'])
#     # means2.append(-1/0.4 * np.log10(gain/t_exp*np.mean(table['aperture_sum']))+ Z_p)
#     means2.append(np.mean(table['rho4_mom']))
#     meanss.append(np.mean(table['n']))
# plt.scatter(means2, meanss, marker = '.')
# plt.xlabel('rho4 moments')
# plt.ylabel('sersic index')

# bsm = model.predict(x = [dic6['features'], dic6['e1_cal'], dic6['anisotropy_corr'], dic6['e2_cal'], dic6['anisotropy_corr_2']])
# path = config.workpath('Test5')
# file_name = os.path.join(path, 'BSM_test3.fits')
# t = Table([bsm], names = ['bsm'], dtype =['f4'])
# t.write(file_name, overwrite = True)
# bsm, bsm1 = np.split(bsm, indices_or_sections = 2,axis = 2)
# table = Table.read(path + '/BSM_test3.fits')
# bsm = table['bsm']
# bsm, bsm1 = np.split(bsm, indices_or_sections = 2,axis = 2)
# plt.scatter(bsm, bsm1)
# means = []
# means2 = []
# params = oneD_to_threeD(path, 'aperture_sum', dic6['n_cas'], dic6['n_rea'])
# for i in range(100):
#     means.append(np.mean(bsm[i][:][0]))
#     means2.append(np.mean(params[i][:][0]))
# plt.scatter(means2, means, marker = '.')
# plt.xlabel('psf polarisation')
# plt.ylabel('$b^{sm}_1$')
#cases_scatter_plot(dic5, bsm1, path2, 'anisotropy_corr_2', 'ML boost-factor', 'anisotropy correction', False)
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




