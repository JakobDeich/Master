import config
from astropy.table import Table
import matplotlib
import tkinter 
matplotlib.use('Tkagg')
# import galsim
import matplotlib.pyplot as plt
# import statistics
import numpy as np
# from matplotlib import cm
from scipy.optimize import curve_fit
import os
from astropy.table import Table, Column, vstack
import time
# import ksb
from pathlib import Path
import pickle

# from ML import cases_scatter_plot
start = time.time()

def linear(x, m, b):
    return m*x + b

def gal_mag(flux):
    return  24.6 - 2.5 * np.log10(3.1/(3*565)* flux)

plt.rcParams.update({'font.size': 14})

# def mean(param,path, corr = False):
#     print('start')
#     means = []
#     means2 = []
#     t = Table.read(config.workpath(path) + '/Measured_ksb_1.fits')
#     ncas = t.meta['N_CAS']
#     for i in range(100):
#         tab = Table.read(config.workpath(path) + '/Measured_ksb_'+ str(i) + '.fits')
#         if corr == True:
#             means.append(np.mean(tab['e1_cal']))
#             means2.append(np.mean(tab['e1_cal']-tab['anisotropy_corr']))
#         else:
#             means.append(np.mean(tab[param]))
#     print('done')
#     return means, means2

# def mean2(param,path, corr = False):
#     print('start')
#     means = []
#     means2 = []
#     t = Table.read(config.workpath(path) + '/Measured_ksb_1.fits')
#     ncas = t.meta['N_CAS']
#     for i in range(100):
#         tab = Table.read(config.workpath(path) + '/Measured_ksb_'+ str(i) + '.fits')
#         if corr == True:
#             means.append(np.mean(tab['e2_cal']))
#             means2.append(np.mean(tab['anisotropy_corr_2']))
#         else:
#             means.append(np.mean(tab[param]))
#     print('done')
#     return means, means2

def mean(param, path = 'Test5'):
    res = []
    for i in range(100):
        tab = Table.read(config.workpath(path) + '/Measured_ksb_'+ str(i) + '.fits')
        if (min(tab['e1_cal']) > -9):
            res.append(np.mean(tab[param]))
    return np.array(res)

def stdaw(param, path = 'Test5'):
    mean = []
    res = []
    for i in range(100):
        means = []
        tab = Table.read(config.workpath(path) + '/Measured_ksb_'+ str(i) + '.fits')
        mean = np.mean(tab[param])
        for j in range(1700):
            means.append(tab[param][j*120:(j+1)*120])
        stabw = np.sum((means - mean)**2)/1700
        res.append(stabw)
    return res
    
    


def mean_3d(array):
    means=[]
    for i in range(len(array)):
        means.append(np.mean(array[i]))
    return means

# tab = Table.read(config.workpath('Test4') + '/Measured_ksb.fits')
# print(max(tab['mag']))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# path = 'Test5'
# BSM_train = Table.read(config.workpath('Test5' + '/BSM_1203_val.fits'))
# print('be')
# bsm1 = mean_3d(BSM_train['bsm_val1'])
# print('ba')
# param = mean('aperture_sum', path)
# param2 = mean('sigma_mom', path)*0.02
# param3 = mean('mag')
# t_exp = 3*565 #s
# gain = 3.1 #e/ADU
# Z_p = 24.6
# param = -1/0.4 * np.log10(gain/t_exp*param)+ Z_p
# print('bu')
# sersic = mean('rho4_mom', path)
# print('bab')
# plt.scatter(param, param3, c = param2, cmap = 'viridis', marker = '.')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.xlabel('magnitude estmiate')
# plt.ylabel('true magnitude')
# plt.colorbar(label = 'half light radius in arcsec')
# plt.scatter(param, bsm1, marker = '.', c = sersic, cmap = 'viridis')
# plt.xlabel('magnitude')
# plt.ylabel('average $b^{sm}_1$')
# plt.colorbar(label = r'$\rho_4$ moments')
def bsm_dep(param_name, train = True, first_comp = True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if train == False:
        name1 = 'val'
        path = 'Test5_validation'
        BSM_train = Table.read(config.workpath('Test5' + '/BSM_1203_val.fits'))
        if first_comp == True:
            bsm = mean_3d(BSM_train['bsm_val1'])
            ylabel = 'average $b^{sm}_1$'
            name2 = 1
        else:
            bsm = mean_3d(BSM_train['bsm_val2'])
            ylabel = 'average $b^{sm}_2$'
            name2 = 2
    else:
        name1 ='train'
        path = 'Test5'
        BSM_train = Table.read(config.workpath('Test5' + '/BSM_1203_train.fits'))
        if first_comp == True:
            bsm = mean_3d(BSM_train['bsm1'])
            ylabel = 'average $b^{sm}_1$'
            name2 = 1
        else:
            bsm = mean_3d(BSM_train['bsm2'])
            ylabel = 'average $b^{sm}_2$'
            name2 = 2
    param = mean(param_name, path)
    if param_name == 'sigma_mom':
        param = param*0.02
        xlabel = 'Galaxy size in arcsec'
        name = 'sigma'
    elif param_name == 'aperture_sum':
        t_exp = 3*565 #s
        gain = 3.1 #e/ADU
        Z_p = 24.6
        param = -1/0.4 * np.log10(gain/t_exp*param)+ Z_p
        xlabel = 'magnitude estimate'
        name = 'apersum'
    sersic = mean('rho4_mom', path)
    plt.scatter(param, bsm, marker = '.', c = sersic, cmap = 'viridis')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label = r'$\rho_4$ moments')
    plt.savefig('Thesis_plots/bsm_' + name + '_' + name1 + str(name2) + '.pdf')
    plt.clf()
    print('done')
    return None

#bsm_dep('sigma_mom', train = True, first_comp=(True))
#bsm_dep('sigma_mom', train = True, first_comp=(False))
#bsm_dep('sigma_mom', train = False, first_comp=(True))
#bsm_dep('sigma_mom', train = False, first_comp=(False))
#bsm_dep('aperture_sum', train = True, first_comp=(True))
#bsm_dep('aperture_sum', train = True, first_comp=(False))
#bsm_dep('aperture_sum', train = False, first_comp=(True))
#bsm_dep('aperture_sum', train = False, first_comp=(False))

# sn = []
# sn2 = []
# for i in range(100):
#     tab = Table.read(config.workpath('Test5')+ '/Measured_ksb_' + str(i) + '.fits')
#     if np.mean(tab['Signal_to_Noise']) < 10:
#         sn.append(np.mean(tab['Signal_to_Noise']))
#     else:
#         sn2.append(np.mean(tab['Signal_to_Noise']))
# for i in range(100):
#     tab = Table.read(config.workpath('Test5')+ '/Measured_ksb_' + str(i) + '.fits')
#     sn.append(np.mean(tab['Signal_to_Noise']))
#     sn2.append(np.mean(tab['mag']))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(sn, sn2)
# c = []
# for i in range(30):
#     if (min(sn) + (max(sn)-min(sn))/30*i) < 10:
#         s = ['r']
#         c.extend(s)
#     else:
#         s = ['b']
#         c.extend(s)
# sn = [sn, sn2]
# plt.hist(sn, bins = 18, density = False, stacked = True, color = ['r', 'b'], label = ['S/N < 10', 'S/N > 10'])
# # plt.hist(sn2,density = True, color = 'b')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.xlabel('Signal to Noise ratio')
# plt.ylabel('counts')
# plt.legend()
# plt.show()

# for i in range(100):
#     tab = Table.read(config.workpath('Test5')+ '/Measured_ksb_' + str(i) + '.fits')
#     # if np.mean(tab['sigma_mom'])*0.02 < np.mean(tab['sigma_mom_psf'])*0.02*1.25:
#     print(np.mean(tab['sigma_mom'])/np.mean(tab['sigma_mom_psf']))

# dic = pickle.load( open(config.workpath('Test5/') + "Processed_data_withM_withSN.p" , "rb" ) )
# print(dic['n_cas'])
# dic = pickle.load( open(config.workpath('Test5/') + "Processed_data_mergedNNcheck_SN0_val.p" , "rb" ) )
# print(dic['n_cas'], dic['c-bias_1'], dic['c-bias_2'])
tab5 = Table.read(config.workpath('Test5') + '/SN10_0604.fits')
tab1 = Table.read(config.workpath('Test5') + '/SN0_noM_0604.fits')
tab3 = Table.read(config.workpath('Test5') + '/SN0_0604.fits')
tab2 = Table.read(config.workpath('Test5') + '/No_Split_0404.fits')
tab6 = Table.read(config.workpath('Test5') + '/No_Split_SN10_0704.fits')
tab4 = Table.read(config.workpath('Test5') + '/No_Split_SN0_0904.fits')
def meanstw(tab):
    #print('mean after training: ' + str(np.mean(tab['train'])) + ' and validation: ' + str(np.mean(tab['val'])) + ' with cases ' + str(len(tab)))
    print(min(tab['train']), min(tab['val']))
    #print((tab['train']), (tab['val']))
    stabw = np.sqrt(np.sum((tab['train']-np.mean(tab['train']))**2)/(len(tab)-1))/np.sqrt(len(tab))
    stabw_val = np.sqrt(np.sum((tab['val']-np.mean(tab['val']))**2)/(len(tab)-1))/np.sqrt(len(tab))
   # print('uncertainty of training: ' + str(stabw) + ' and validation: ' + str(stabw_val) + '\n')
    return None

meanstw(tab1)
meanstw(tab2)
meanstw(tab3)
meanstw(tab4)
meanstw(tab5)
meanstw(tab6)
    
# print(tab['train'], tab['val'])
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(mean('e2_cal_psf'), mean('M4_2_psf_adap'))
# popt, pcov = curve_fit(linear,mean('e2_cal_psf'),mean('M4_2_psf_adap'))
# plt.plot(mean('e2_cal_psf'),linear(mean('e2_cal_psf'), popt[0], popt[1]), color='r')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.xlabel('$e_2^{PSF}$')
# plt.ylabel('$Q^{(4)}_2$')
# plt.tight_layout()
# print(stdaw('M4_1_psf_adap'))
# param1 = []
# param2 = []
# for i in range(100):
#     tab = Table.read(config.workpath('Test5') + '/Measured_ksb_' + str(i) + '.fits')
#     param1.append(np.mean(tab['n']))
#     # t_exp = 3*565 #s
#     # gain = 3.1 #e/ADU
#     # Z_p = 24.6
#     # l_pix = 0.1 #arcsec
#     # param2.append(np.mean(1/(-0.4)*np.log10(gain/t_exp*tab['aperture_sum'])+Z_p))
#     param2.append(np.mean(tab['rho4_mom']))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(param2, param1, marker='.')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.grid(True)
# plt.ylabel('Sérsic index')
# plt.xlabel(r'$\rho_4$ moments')
# tab = Table.read(config.workpath('M4_Check') + '/Measured_ksb.fits')
# plt.scatter(mean('e2_cal_psf'), mean('M4_2_psf_adap'))
# plt.xlabel('psf polarisation $e_2$')
# plt.ylabel('$M^{(4)}_2$')
# plt.tight_layout()
# # plt.axis('equal')

# path = config.workpath('Test5')
# path2 = config.workpath('Test5_validation')
# # print('start')
# # dic_train = pickle.load( open(path + "/Processed_data_new.p", "rb" ) )
# # print('go')
# # print(table)
# #cases_scatter_plot(dic_train, table['bsm2'], 'Test5', 'M4_1_psf_adap', 'ML-boosted $b^{sm}$', 'fourth moments $M^{(4)}_1$', False)
# path = config.workpath('Test5')
# table = Table.read(path + '/Ms_0603.fits')
# print(min(np.sum(table['Ms'], axis =1)))
# table = Table.read(path + '/BSM_1203_train.fits')
# print(table)
# table2 = Table.read(path + '/BSM_0303_val.fits')
# dic_train = pickle.load( open(path + "/Processed_data_noM.p", "rb" ) )
# dic_val = pickle.load( open(path2 + "/Processed_data_noM.p", "rb" ) )
# # print(dic_train['features'][:,:,1])
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(mean_3d(dic_train['features'][:,:,3]*40),mean_3d(table['bsm1']), marker = '.')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.xlabel('PSF size')
# plt.ylabel('$b^{sm}_1$')
# plt.savefig('Thesis_plots/bsm_rho4_train1.pdf')
# plt.clf()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(mean_3d(dic_train['features'][:,:,2]*2.6),mean_3d(table['bsm2']), marker = '.')
# ax.set_aspect(1.0/ax.get_data_ra# table = Table.read(path + '/BSM_1203_train.fits')
# table2 = Table.read(path + '/BSM_1203_val.fits')tio(), adjustable='box')
# plt.xlabel(r'$\rho_4$ moments')
# plt.ylabel('$b^{sm}_2$')
# plt.savefig('Thesis_plots/bsm_rho4_train2.pdf')
# plt.clf()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(mean_3d(dic_val['features'][:,:,2]*2.6),mean_3d(table2['bsm_val1']), marker = '.')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.xlabel(r'$\rho_4$ moments')
# plt.ylabel('$b^{sm}_1$')
# plt.savefig('Thesis_plots/bsm_rho4_val1.pdf')
# plt.clf()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(mean_3d(dic_val['features'][:,:,2]*2.6),mean_3d(table2['bsm_val2']), marker = '.')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.xlabel(r'$\rho_4$ moments')
# plt.ylabel('$b^{sm}_2$')
# plt.savefig('Thesis_plots/bsm_rho4_val2.pdf')
# plt.clf()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# bsm1, bsm2 = np.split(table['bsm'],2)
# plt.scatter(mean_3d(bsm1),mean_3d(bsm2), marker = '.')
# plt.xlabel('$b^{sm}_1$')
# plt.ylabel('$b^{sm}_2$')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.tight_layout()
# plt.savefig('Thesis_plots/BSMs_tog.pdf')
# plt.clf()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(mean_3d(table2['bsm_val1']),mean_3d(table2['bsm_val2']), marker = '.')
# plt.xlabel('$b^{sm}_1$')
# plt.ylabel('$b^{sm}_2$')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.tight_layout()
# plt.savefig('Thesis_plots/BSMs_tog_val.pdf')

# print(mean('anisotropy_corr'), mean_3d(table['bsm1']))
# means= mean('sigma_mom',path)
# gain = 3.1 #e/ADU
# Z_p = 24.6
# l_pix = 0.1 #arcsec
# t_exp = 3*565 #s
# fig = plt.figure()
# ax = fig.add_subplot(111)
# means = means*0.02
# plt.scatter(means, mean_3d(table['bsm1']), marker = '.')
# # plt.scatter(means2, mean_3d(table['bsm1']), label = '$e_1 - \delta e_1$', marker = '.')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.ylabel('average $b^{sm}_1$')
# # plt.xlabel('anisotropy correction $\delta e$')
# plt.xlabel('Galaxy size in arcsec')
# plt.tight_layout()
# # plt.legend()
# plt.savefig('Thesis_plots/bsm_sigma_train1.pdf')
# plt.clf()
# means1, means2 = mean2('blue', path, corr = True)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # plt.scatter(means1, mean_3d(table['bsm2']), label ='$e_2$', marker = '.')
# plt.scatter(means2, mean_3d(table['bsm2']), label = '$e_2 - \delta e_2$', marker = '.')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.ylabel('average $b^{sm}_2$')
# plt.xlabel('anisotropy correction $\delta e$')
# # plt.xlabel('polarisation')
# #plt.legend()
# plt.savefig('Thesis_plots/bsm_ac_2.pdf')
# plt.clf()
# means1, means2 = mean2('blue', path2, corr = True)
# plt.scatter(means1, mean_3d(table2['bsm_val2']), label ='$e_2$', marker = '.')
# plt.scatter(means2, mean_3d(table2['bsm_val2']), label = '$e_2 - \delta e_2$', marker = '.')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.ylabel('average $b^{sm}_2$')
# plt.xlabel('anisotropy correction $\delta e$')
# plt.legend()
# plt.savefig('Thesis_plots/bsm_ac_val2.pdf')
# plt.clf()
# means1, means2 = mean('blue',path2, corr = True)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(means1, mean_3d(table2['bsm_val1']), label ='$e_1$', marker = '.')
# plt.scatter(means2, mean_3d(table2['bsm_val1']), label = '$e_1 - \delta e_1$', marker = '.')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.ylabel('average $b^{sm}_1$')
# plt.xlabel('anisotropy correction $\delta e$')
# plt.legend()
# plt.savefig('Thesis_plots/bsm_pol_val1.pdf')
# plt.clf()
# plt.tight_layout()

# path = config.workpath('Test3')
# cases = [0,20,30,40,50, 60, 70]
# trys =np.arange(15)
# final = []
# for j in cases:
#     for i in trys:
#         image_file = path + '/Grid_case' + str(j) +'_' + str(i) + '_3400.fits'
#         image_file = Path(image_file)
#         if image_file.exists():
#             True
#         else:
#             param = [j,i]
#             final.append(param)n Dozent werde ich die Tage noch 
# print(final)
            

# path = config.workpath('Test3')
# cases = [0,20,30,40,50, 60, 70]
# # cases = [5,10,15]
# # trys = np.arange(25)
# trys = np.arange(15)
# st = []
# nreas = ['_200','_400', '','_1700', '_3400']
# reas = [200*120,400*120,800*120, 1700*120, 3400*120]
# for j in cases:
#     sts = []
#     for rea in nreas:
#         pol = []
#         param = []
#         for i in trys:
#             file_name = path + '/Measured_ksb_' + str(j) +'_' + str(i) + rea + '.fits'
#             file_name = Path(file_name)
#             if file_name.exists():
#                 table = Table.read(file_name)
#                 if min(table['e1_cal'] > -9):
#                     pol.append(np.mean(table['e1_cal']-table['anisotropy_corr']))
#                     param.append(np.mean(table['aperture_sum']))
#                 else:
#                     print('oops')
#             else:
#                 print(j, i, rea)
#         pol_mean = np.mean(pol)
#         stabw = np.sum((pol-pol_mean)**2)/len(pol)
#         stabw = np.sqrt(stabw)
#         sts.append(stabw*10**4)
#         # plt.scatter(param,pol, marker = '.', label = 'case ' + str(j))
#         # plt.errorbar(np.mean(param), pol_mean, yerr = stabw, fmt = 'x', )
#     st.append(sts)
# for i in range(len(cases)):
#     plt.plot(reas, st[i], marker = 'o',label = 'case: ' + str(cases[i]))
# plt.legend()
# plt.xlabel('number of galaxy stamps')
# plt.ylabel('standard deviation of additive bias [$10^{-4}$]')
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.tight_layout
# mydir = config.workpath('Test3')
# es = []
# for i in range(100):
#     table1 = Table.read(mydir + '/Measured_ksb_'+ str(i) + '.fits')
#     e_pol = np.mean(table1['e1_cal_psf'])
#     if e_pol > 0.05 and e_pol < 2:
#         es.append(e_pol)
# print(np.mean(es), len(es))
    



# table = Table.read(mydir + '/Input_data_120.fits')
# print(np.mean(table['mag']))
# table = Table(names = ['mag', 'n', 'r_half', 'e1', 'e2','Q111_psf','Q222_psf', 'gamma1', 'b_sm'], dtype = ['f4', 'f4', 'f4', 'f4','f4','f4', 'f4', 'f4', 'f4'])
# for i in range(200):
#     table1 = Table.read(mydir + '/Measured_ksb_'+ str(i) + '.fits')
#     table = vstack([table,table1]) 
# if not os.path.isdir(mydir):
#     os.mkdir(mydir)
# file_name = os.path.join(mydir, 'Measured_ksb.fits')
# table.write( file_name , overwrite=True)     

# table = Table.read(mydir + '/Input_data_5.fits')
# ksb.calculate_ksb_training('Test2', 5)
# print(table)

# b = 1
# gamma_cal = []
# gamma_real = np.linspace(-0.1,0.1,20)
# bsm = [1,1.3]
# cs = []
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for b in bsm:
#     gamma_cal = []
#     for i in range(20):
#         path = config.workpath('Run6/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
#         table = Table.read(path)
#         e1 = table['e1_cal']
#         e_corr = table['anisotropy_corr']
#         P_g11 = table['Pg_11']
#         mean1 = np.mean(e1-b*e_corr)
#         mean2 = np.mean(P_g11)
#         gamma_cal.append(mean1/mean2 - gamma_real[i])
#     popt, pcov = curve_fit(linear, gamma_real, gamma_cal)
#     plt.scatter(gamma_real, gamma_cal, label = '$b^{sm}_1$ = ' + str(b) + ' , $c_1$ = ' + str(round(popt[1], 4)))
#     plt.plot(gamma_real, linear(gamma_real, popt[0], popt[1]))
#     cs.append(popt[1]) 
# popt, pcov = curve_fit(linear, bsm, cs)
# # plt.scatter(bsm, cs)
# # plt.plot(bsm, linear(bsm, popt[0], popt[1]))
# m2 = popt[0]
# b2 = popt[1]
# # plt.scatter(-1*b2/m2, 0, s = 80, color = 'r', marker = 'D',label = 'target boost factor')    
# # plt.xlabel('boost factor $b^{sm}_1$')
# # plt.ylabel('additive bias $c_1$')
# plt.xlabel('$g_1^{true}$')
# plt.ylabel('$g_1^{measured} - g_1^{true}$')
# plt.legend()
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
# plt.savefig('Plots2/Boost_determine_Schema.pdf')
# print(popt[1])

    


# shear = np.linspace(-0.06, 0.06, 5)
# gamma_cal = []
# gamma_real = np.linspace(-0.06, 0.06, 5)
# for j in shear:
#     g1 = []
#     for Galaxy in table:
#         if (Galaxy['gamma1']-10**(-4)) <= j and (Galaxy['gamma1']+10**(-4)) >= j:
#             param = Galaxy['g1_cal_galsim']
#             g1.append(param)
#     g1 = np.asarray(g1)
#     mean1 = np.mean(g1)
#     gamma_cal.append(mean1)
# popt, pcov = curve_fit(linear, gamma_real, gamma_cal)
# plt.scatter(gamma_real, gamma_cal)
# plt.show()


# mydir = config.workpath('Test3')
# table = Table.read(mydir + '/Measured_ksb.fits')
# plt.scatter(table['mag'], table['aperture_sum'])
# plt.savefig('Plots2/mag_apersum.png')

def determine_boost_quick(path, case):
    print('determining boost factor for case ' + str(case))
    mydir = config.workpath(path)
    # table = Table.read(mydir + '/Measured_ksb_' + str(case) + '.fits')
    table = Table.read(mydir + '/Measured_ksb.fits')
    # for Galaxy in table:
    #     print(Galaxy['psf_pol'], Galaxy['anisotropy_corr'], gal_mag(Galaxy['aperture_sum']))
    #     break
    shears = table.meta['N_SHEAR']
    n_rea = table.meta['N_REA']
    n_canc = table.meta['N_CANC']
    bsm = np.linspace(1.,1.4,3)
    cs = []
    for b in bsm:
        e_corr = []
        P_g11 = []
        e1 = []
        for Galaxy in table:
            # if Galaxy['Flag'] == 0:
                param = Galaxy['e1_cal']
                e1.append(param)
                param = Galaxy['anisotropy_corr']
                e_corr.append(param)
                param = Galaxy['Pg_11']
                P_g11.append(param)
        e1 = np.asarray(e1)
        e_corr = np.asarray(e_corr)
        P_g11 = np.asarray(P_g11)
        mean1 = np.mean(e1- b * e_corr)
        mean2 = np.mean(P_g11)
        cs.append(mean1/mean2)
        # popt, pcov = curve_fit(linear, gamma_real, gamma_cal)
        # bias = popt[1]
        # cs.append(np.mean(gamma_cal)) 
    popt, pcov = curve_fit(linear, bsm, cs)
    m2 = popt[0]
    b2 = popt[1]
    print(-1*b2/m2)
    # Col_A = Column(name = 'b_sm', data = tab)
    # try:
    #     table.add_columns([Col_A])
    # except:
    #     table.replace_column(name = 'b_sm', col = Col_A)
    # table.write( mydir + '/Measured_ksb_' + str(case) + '.fits' , overwrite=True) 
    return None

def determine_boost(path, case):
    print('determining boost factor for case ' + str(case))
    mydir = config.workpath(path)
    table = Table.read(mydir + '/Measured_ksb.fits')
    # for Galaxy in table:
    #     print(Galaxy['psf_pol'], Galaxy['anisotropy_corr'], gal_mag(Galaxy['aperture_sum']))
    #     break
    shears = table.meta['N_SHEAR']
    n_rea = table.meta['N_REA']
    n_canc = table.meta['N_CANC']
    bsm = np.linspace(1.,1.4,3)
    gamma = np.linspace(-0.01,0.01,3)
    cs = []
    for b in bsm:
        gamma_cal = []
        for j in gamma:
            e_corr = []
            P_g11 = []
            e1 = []
            for Galaxy in table:
                if (Galaxy['gamma1']-10**(-4)) <= j and (Galaxy['gamma1']+10**(-4)) >= j and Galaxy['Flag'] == 0:
                    param = Galaxy['e1_cal']
                    e1.append(param)
                    param = Galaxy['anisotropy_corr']
                    e_corr.append(param)
                    param = Galaxy['Pg_11']
                    P_g11.append(param)
            e1 = np.asarray(e1)
            e_corr = np.asarray(e_corr)
            P_g11 = np.asarray(P_g11)
            mean1 = np.mean(e1- b * e_corr)
            mean2 = np.mean(P_g11)
            gamma_cal.append(mean1/mean2)
        popt, pcov = curve_fit(linear, gamma, gamma_cal)
        bias = popt[1]
        cs.append(bias) 
    popt, pcov = curve_fit(linear, bsm, cs)
    m2 = popt[0]
    b2 = popt[1]
    print(-1*b2/m2)
    # Col_A = Column(name = 'b_sm', data = tab)
    # try:
    #     table.add_columns([Col_A])
    # except:
    #     table.replace_column(name = 'b_sm', col = Col_A)
    # table.write( mydir + '/Measured_ksb_' + str(case) + '.fits' , overwrite=True) 
    return None


# determine_boost_quick('Test3',3)
# mydir = config.workpath('Test3')
# determine_boost_quick(mydir , 0)
# determine_boost(mydir , 0)
# # table = Table.read(mydir + '/Measured_ksb.fits')
# table = Table.read(mydir + '/Input_data_17.fits')
# print(table)
# # bsm = []
# # for i in range(200):
# #     bsm.append(table['b_sm'][i*10000])
# # print(bsm)
# # print(table['mag'][6*500])
# bsm = []
# psfs = []
# mags = []
# rs = []
# for i in range(197):
#     bsm.append(table['b_sm'][i*10000])
#     mags.append(table['psf_pol'][i*10000])
#     rs.append(table['r_half'][i*10000])
# # # print(mags, rs)
# # plt.scatter(mags,rs, c = bsm , marker = 'o', cmap = 'bwr')
# # plt.colorbar()
# # plt.xlabel('psf polarisation')
# # plt.ylabel('half light radius')
# plt.hist(bsm, bins = 50, range= (0,4))
# plt.xlabel('boost factor $b_{sm}$')
# plt.ylabel('counts')
# plt.savefig('Plots2/Histo_boost_big_new.pdf')


def Bootstrap_boost(table,bin_digit,bin_num, b):
    Error = 0
    gamma_errs = []
    Good = bin_digit == bin_num
    size = sum(Good)
    numbers = table['number'][Good]
    for i in range(200):
        rng = np.random.default_rng()
        numb_random = rng.choice(numbers, size = (size))
        e1_err1 = table['e1_cal'][2*numb_random-2]
        e1_err2 = table['e1_cal'][2*numb_random-1]
        e1_err = np.concatenate((e1_err1, e1_err2))
        anis_err1 = table['anisotropy_corr'][2*numb_random-2]
        anis_err2 = table['anisotropy_corr'][2*numb_random-1]
        anis_err = np.concatenate((anis_err1, anis_err2))
        Pg_err2 = table['Pg_11'][2*numb_random-2]
        Pg_err1 = table['Pg_11'][2*numb_random-1]
        Pg_err = np.concatenate((Pg_err1, Pg_err2))
        Flags1 = table['Flag'][2*numb_random-2]
        Flags2 = table['Flag'][2*numb_random-1]  
        Flags = np.concatenate((Flags1, Flags2))
        Is_good = Flags == 0
        pol_err = np.mean(e1_err[Is_good] - b* anis_err[Is_good])
        Pg_err = np.mean(Pg_err[Is_good])
        gamma_errs.append(pol_err/Pg_err)
    Error = np.sqrt(np.mean((gamma_errs-np.mean(gamma_errs))**2))
    return Error
        
        
def Bootstrap(table, b, flag = True):
    Error = 0
    gamma_errs = []
    if flag == True:
        for i in range(200):
            # if i%25==0:
            #     print(i)
            rng = np.random.default_rng()
            numb_random = rng.choice(range(table.meta['NY_TILES'] * table.meta['NX_TILES']), size = (table.meta['NY_TILES'] * table.meta['NX_TILES']))
            e1_err1 = table['e1_cal'][2*numb_random-2]
            e1_err2 = table['e1_cal'][2*numb_random-1]
            e1_err = np.concatenate((e1_err1, e1_err2))
            anis_err1 = table['anisotropy_corr'][2*numb_random-2]
            anis_err2 = table['anisotropy_corr'][2*numb_random-1]
            anis_err = np.concatenate((anis_err1, anis_err2))
            Pg_err2 = table['Pg_11'][2*numb_random-2]
            Pg_err1 = table['Pg_11'][2*numb_random-1]
            Pg_err = np.concatenate((Pg_err1, Pg_err2))
            Flags1 = table['Flag'][2*numb_random-2]
            Flags2 = table['Flag'][2*numb_random-1]  
            Flags = np.concatenate((Flags1, Flags2))
            Is_good = Flags == 0
            pol_err = np.mean(e1_err[Is_good] - b* anis_err[Is_good])
            Pg_err = np.mean(Pg_err[Is_good])
            gamma_errs.append(pol_err/Pg_err)
        Error = np.sqrt(np.mean((gamma_errs-np.mean(gamma_errs))**2))
    else:
        for i in range(200):
            rng = np.random.default_rng()
            numb_random = rng.choice(range(table.meta['NY_TILES'] * table.meta['NX_TILES']), size = (table.meta['NY_TILES'] * table.meta['NX_TILES']))
            e1_err1 = table['e1_cal'][2*numb_random-2]
            e1_err2 = table['e1_cal'][2*numb_random-1]
            e1_err = np.concatenate((e1_err1, e1_err2))
            anis_err1 = table['anisotropy_corr'][2*numb_random-2]
            anis_err2 = table['anisotropy_corr'][2*numb_random-1]
            anis_err = np.concatenate((anis_err1, anis_err2))
            Pg_err2 = table['Pg_11'][2*numb_random-2]
            Pg_err1 = table['Pg_11'][2*numb_random-1]
            Pg_err = np.concatenate((Pg_err1, Pg_err2))
            pol_err = np.mean(e1_err - b* anis_err)
            Pg_err = np.mean(Pg_err)
            gamma_errs.append(pol_err/Pg_err)
        Error = np.sqrt(np.mean((gamma_errs-np.mean(gamma_errs))**2))  
    return Error

def Filter(table, Filter_Var, Sorting_Var='' , bin_digit = [], Sort = False, bin_num = 0):
    if Sort == False:
        Sorting_Var = Filter_Var
    Filter_Var1 = table[Filter_Var]
    Vars2 = table[Sorting_Var]
    if Sorting_Var == 'aperture_sum':
        Vars2 = 24.6 - 2.5 * np.log10(3.1/(3*565)* Vars2)
    if Filter_Var == 'aperture_sum':
        Filter_Var1 = 24.6 - 2.5 * np.log10(3.1/(3*565)* Filter_Var1)
    Good2 = np.isnan(table['rho4_mom'])
    Good2 = np.logical_not(Good2)
    Good2 = np.array([Good2]).transpose()
    Good3 = table['aperture_sum'] > 600
    Good2 = np.logical_and(Good2, Good3)
    Good1 = table['sigma_mom'] > 0
    Good1 = np.array([Good1]).transpose()
    Good = np.logical_and(Good1, Good2)
    Good = Good[:,0]
    if Sort == True:
        Good1 = bin_digit == bin_num
        if Sorting_Var == 'aperture_sum':
            Good1 = Good1[:,0]
        Good = np.logical_and(Good, Good1)
    Filter_Var1 = Filter_Var1[Good]
    return Filter_Var1
 

# a = [-1,2,4,650,np.log(-1)]   
# table = Table([a], names = ['sigma_mom'], dtype = ['f4'])


# q = np.linspace(0,1,6)
# Good = q > 0
# q = q[Good]
# path = config.workpath('Run6/PSF_es_5/Measured_ksb.fits')
# table = Table.read(path)
# ns = table['n']
# rho4 = table['rho4_mom']
# Good = rho4 > 0
# plt.scatter(ns[Good], rho4[Good])
# plt.xlabel('sersic index n GEMS')
# plt.ylabel('rho4')
# plt.savefig('Plots2/rho4_sersic.pdf')
# Vars1 = Filter(table, 'r_half')
# bins = np.quantile(table['r_half'], q = q)
# bin_dig = np.digitize(table['r_half'], bins)
# print(bin_dig, bins, q)
# for j in range(5):
#     Var = Filter(table, 'sigma_mom', 'r_half', bin_dig ,True , j)
#     print(np.mean(Var), len(Var))
# min_var = min(Var)
# max_var = max(Var)
# print(min_var, max_var)
# step = (max_var-min_var)/6
# print(Filter(table, 'anisotropy_corr','aperture_sum', True, min_var, step, j=1))

def boostFactorDep(Var,Var_initial, Var_name, N):
    q = np.linspace(0,1,N+1)
    Good = q > 0
    q = q[Good]
    #for j in range(N):
        # for i in range(20):
        #     path = config.workpath('Run6/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
        #     table1 = Table.read(path)
        #     Vars1 = Filter(table1, Var)
        #     min_var1 = min(Vars1)
        #     max_var1 = max(Vars1)
        #     mins.append(min_var1)
        #     maxs.append(max_var1)
    path = config.workpath('Run6/PSF_es_20/Measured_ksb.fits')
    table1 = Table.read(path)
    Vars1 = Filter(table1, Var_initial)
    # min_var = min(mins)
    # max_var = max(maxs)
    # print(min_var, max_var)
    bins = np.quantile(table1[Var_initial], q = q)
    bin_dig = np.digitize(table1[Var_initial], bins)
    print(bins, bin_dig)
    # step = (max_var-min_var)/N
    Var_plot = Table(names =['means', 'size', 'bin'], dtype = ['f4', 'i4', 'i4'])
    bsm = np.linspace(1.1,1.4,3)
    gamma_real = np.linspace(-0.1, 0.1, 20)
    bs_err = []
    bs_real = []
    for j in range(N):
        cs = []
        cs_err = []
        for b in bsm:
            gamma_cal = []
            Errors = []
            for i in range(20):
                path = config.workpath('Run6/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
                table = Table.read(path)
                e1 = Filter(table, 'e1_cal',Var_initial, bin_dig ,True , j)
                e_corr = Filter(table, 'anisotropy_corr',Var_initial, bin_dig ,True , j)
                P_g11 = Filter(table, 'Pg_11',Var_initial, bin_dig ,True , j)
                Var_t = Filter(table, Var_initial,Var_initial, bin_dig ,True , j)
                params = [np.mean(Var_t), len(Var_t), j]
                Var_plot.add_row(params)
                mean1 = np.mean(e1-b*e_corr)
                mean2 = np.mean(P_g11)
                gamma_cal.append(mean1/mean2)
                Errors.append(Bootstrap_boost(table, bin_dig, j, b))
            popt, pcov = curve_fit(linear, gamma_real, gamma_cal, sigma = Errors, absolute_sigma = True)
            m = popt[0]
            bias = popt[1]
            cs.append(bias) 
            cs_err.append(np.sqrt(pcov[1,1]))
        popt, pcov = curve_fit(linear, bsm, cs, sigma = cs_err, absolute_sigma = True)
        m2 = popt[0]
        b2 = popt[1]
        bs_real.append(-1*b2/m2)
        bs_err.append(np.sqrt(np.sqrt((1/m2*pcov[1,1])**2 + (b2/m2**2*pcov[0,0])**2)))
        print('boost factor determined with: ' + str(-1*b2/m2) + ' and ' + str(np.sqrt(np.sqrt((1/m2*pcov[1,1])**2 + (b2/m2**2*pcov[0,0])**2))))
    Var_plots = []
    for bins in range(N):
        Good = Var_plot['bin']==bins
        size = sum(Var_plot['size'][Good])
        weighted_mean = sum(Var_plot['means'][Good]*Var_plot['size'][Good])
        Var_plots.append(weighted_mean/size)
    plt.errorbar(Var_plots, bs_real, yerr = bs_err, ecolor = 'red', fmt = 'r.')
    plt.xlabel(Var_name)
    plt.ylabel('$b^{sm}_1$')
    plt.savefig('Plots2/boost_' + Var_name + '.pdf')
    return 0

def boostFactorDep_save(Var, Var_name, N):
    try:
        boostFactorDep(Var, Var_name, N)
    except:
        boostFactorDep(Var, Var_name, N-1)
    return 0
        

#boostFactorDep('sigma_mom','r_half', 'sigma moments', 5)
#boostFactorDep('rho4_mom', 'rho4', 5)
#boostFactorDep('aperture_sum','mag', 'magnitude', 5)
#boostFactorDep('mag', 'magnitude GEMS', 5)
#boostFactorDep('r_half', 'half light radius GEMS [arcsec]', 5)

# e1_s = []

# path = config.workpath('Run1/PSF_es_1/Measured_ksb.fits')
# table1 = Table.read(path)
# AS = table1['sigma_mom']
# # AS = 24.6 - 2.5 * np.log10(3.1/(3*565)* AS)
# plt.hist(AS, 10)                          

# path = config.workpath('Run6/PSF_es_1/Measured_ksb.fits')
# table = Table.read(path)
# Good = table['sigma_mom'] > 0
# plt.scatter(table['r_half'][Good], table['sigma_mom'][Good], s=2)
# plt.xlabel('half light radius GEMS [arcsec]')
# plt.ylabel('sigma moment')
# plt.savefig('Plots2/r_half_sigma_mom.pdf')

#smear polarizability in dependence on magnitude
# bsm = np.linspace(1,1.4,8)
# bs_real = []
# mags = [20.5,21.5,22.5,23.5,24.5]
# for mag in mags:
#     cs = []
#     for b in bsm:
#         gamma_cal = []
#         gamma_real = []
#         for i in range(20):
#             path = config.workpath('Run6/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
#             table = Table.read(path)
#             path2 = config.workpath('Run6/PSF_es_' + str(i+1) +'/Gamma.fits')
#             Gamma = Table.read(path2)
#             Mag = table['mag']
#             Mag_range = Mag > (mag-0.5)
#             Mag_range2 = Mag < (mag+0.5)
#             MAG_RANGE = np.logical_and(Mag_range, Mag_range2)
#             e1 = table['e1_cal']
#             e_corr = table['anisotropy_corr']
#             P_g11 = table['Pg_11']
#             #print(P_g11)
#             mean1 = np.mean(e1[MAG_RANGE]-b*e_corr[MAG_RANGE])
#             mean2 = np.mean(P_g11[MAG_RANGE])
#             #mean3 = np.mean(e1)
#             gamma_cal.append(mean1/mean2)
#             for Galaxy in Gamma:
#                 gamma_real.append(Galaxy['gamma1'])
#         z = np.polyfit(gamma_real, gamma_cal, 1)
#         m = z[0]
#         b = z[1]
#         cs.append(b)
#     css = np.polyfit(bsm, cs,1)
#     bsm2 = np.linspace(1,1.4,200)
#     m2 = css[0]
#     b2 = css[1]
#     bs = []
#     for i in bsm2:
#         if abs(linear(i, m2, b2)) < 10**(-5):
#             bs.append(i)
#         elif abs(linear(i, m2, b2)) < 10**(-4):
#             bs.append(i)
#     bs_real.append(np.median(bs))
# plt.plot(mags, bs_real, color = 'black', marker = 'o', linestyle = '')
# plt.xlabel('magnitude')
# plt.ylabel('$b^{sm}$')


# #additive bias in dependence on magnitude
# mags = [20.5,21.5,22.5,23.5,24.5]
# cs_uncorr =[]
# cs = []
# for mag in mags:
#     b = 1
#     gamma_cal = []
#     gamma_real = []
#     for i in range(20):
#         path = config.workpath('Run6/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
#         table = Table.read(path)
#         path2 = config.workpath('Run6/PSF_es_' + str(i+1) +'/Gamma.fits')
#         Gamma = Table.read(path2)
#         Mag = table['mag']
#         Mag_range = Mag > (mag-0.5)
#         Mag_range2 = Mag < (mag+0.5)
#         MAG_RANGE = np.logical_and(Mag_range, Mag_range2)
#         e1 = table['e1_cal']
#         P_g11 = table['Pg_11']
#         e_corr = table['anisotropy_corr']
#         #print(P_g11)
#         mean1 = np.mean(e1[MAG_RANGE] - e_corr[MAG_RANGE])
#         mean2 = np.mean(P_g11[MAG_RANGE])
#         #mean3 = np.mean(e1)
#         gamma_cal.append(mean1/mean2)
#         for Galaxy in Gamma:
#             gamma_real.append(Galaxy['gamma1'])
#     z = np.polyfit(gamma_real, gamma_cal, 1)
#     m = z[0]
#     b = z[1]
#     cs.append(b)

# for mag in mags:
#     b = 1
#     gamma_cal = []
#     gamma_real = []
#     for i in range(20):
#         path = config.workpath('Run6/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
#         table = Table.read(path)
#         path2 = config.workpath('Run6/PSF_es_' + str(i+1) +'/Gamma.fits')
#         Gamma = Table.read(path2)
#         Mag = table['mag']
#         Mag_range = Mag > (mag-0.5)
#         Mag_range2 = Mag < (mag+0.5)
#         MAG_RANGE = np.logical_and(Mag_range, Mag_range2)
#         e1 = table['e1_cal']
#         P_g11 = table['Pg_11']
#         #print(P_g11)
#         mean1 = np.mean(e1[MAG_RANGE])
#         mean2 = np.mean(P_g11[MAG_RANGE])
#         #mean3 = np.mean(e1)
#         gamma_cal.append(mean1/mean2)
#         for Galaxy in Gamma:
#             gamma_real.append(Galaxy['gamma1'])
#     print(len(e1[MAG_RANGE]))
#     z = np.polyfit(gamma_real, gamma_cal, 1)
#     m = z[0]
#     b = z[1]
#     cs_uncorr.append(b)
# plt.plot(mags, cs, 'ro', label = 'KSB')
# plt.plot(mags, cs_uncorr, color = 'orange', marker = 'o', linestyle = '', label = 'uncorrected')
# plt.xlabel('magnitude')
# plt.ylabel('$c_1$')
# plt.legend()


#smear polarizability in dependence on magnitude
# bsm = np.linspace(1,1.6,12)
# bs_real = []
# r_halfs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# for rs in r_halfs:
#     cs = []
#     for b in bsm:
#         gamma_cal = []
#         gamma_real = []
#         for i in range(20):
#             path = config.workpath('Run6/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
#             table = Table.read(path)
#             path2 = config.workpath('Run6/PSF_es_' + str(i+1) +'/Gamma.fits')
#             Gamma = Table.read(path2)
#             RH = table['r_half']
#             r_range = RH > (rs-0.05)
#             r_range2 = RH < (rs+0.05)
#             R_RANGE = np.logical_and(r_range, r_range2)
#             e1 = table['e1_cal']
#             P_g11 = table['Pg_11']
#             e_corr = table['anisotropy_corr']
#             #print(P_g11)
#             mean1 = np.mean(e1[R_RANGE] - b * e_corr[R_RANGE])
#             mean2 = np.mean(P_g11[R_RANGE])
#             #mean3 = np.mean(e1)
#             gamma_cal.append(mean1/mean2)
#             for Galaxy in Gamma:
#                 gamma_real.append(Galaxy['gamma1'])
#         z = np.polyfit(gamma_real, gamma_cal, 1)
#         m = z[0]
#         b = z[1]
#         cs.append(b)
#     css = np.polyfit(bsm, cs,1)
#     bsm2 = np.linspace(1,1.7,300)
#     m2 = css[0]
#     b2 = css[1]
#     bs = []
#     for i in bsm2:
#         if abs(linear(i, m2, b2)) < 10**(-5):
#             bs.append(i)
#         elif abs(linear(i, m2, b2)) < 10**(-4):
#             bs.append(i)
#     bs_real.append(np.median(bs))
# plt.plot(r_halfs, bs_real, color = 'black', marker = 'o', linestyle = '')
# plt.xlabel('$r_h$ [arcsec]')
# plt.ylabel('$b^{sm}$')

# #additive bias in dependence on half light radius
# r_halfs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# cs_uncorr =[]
# cs = []
# for rs in r_halfs:
#     b = 1
#     gamma_cal = []
#     gamma_real = []
    # for i in range(20):
    #     path = config.workpath('Run6/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
    #     table = Table.read(path)
    #     path2 = config.workpath('Run6/PSF_es_' + str(i+1) +'/Gamma.fits')
    #     Gamma = Table.read(path2)
    #     RH = table['r_half']
    #     r_range = RH > (rs-0.05)
    #     r_range2 = RH < (rs+0.05)
    #     R_RANGE = np.logical_and(r_range, r_range2)
    #     e1 = table['e1_cal']
    #     P_g11 = table['Pg_11']
    #     e_corr = table['anisotropy_corr']
    #     #print(P_g11)
    #     mean1 = np.mean(e1[R_RANGE] - e_corr[R_RANGE])
    #     mean2 = np.mean(P_g11[R_RANGE])
    #     #mean3 = np.mean(e1)
    #     gamma_cal.append(mean1/mean2)
    #     for Galaxy in Gamma:
    #         gamma_real.append(Galaxy['gamma1'])
#     z = np.polyfit(gamma_real, gamma_cal, 1)
#     m = z[0]
#     b = z[1]
#     cs.append(b)

# for rs in r_halfs:
#     b = 1
#     gamma_cal = []
#     gamma_real = []
#     for i in range(20):
#         path = config.workpath('Run6/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
#         table = Table.read(path)
#         path2 = config.workpath('Run6/PSF_es_' + str(i+1) +'/Gamma.fits')
#         Gamma = Table.read(path2)
#         RH = table['r_half']
#         r_range = RH > (rs-0.05)
#         r_range2 = RH < (rs+0.05)
#         R_RANGE = np.logical_and(r_range, r_range2)
#         e1 = table['e1_cal']
#         P_g11 = table['Pg_11']
#         e_corr = table['anisotropy_corr']
#         #print(P_g11)
#         mean1 = np.mean(e1[R_RANGE])
#         mean2 = np.mean(P_g11[R_RANGE])
#         #mean3 = np.mean(e1)
#         gamma_cal.append(mean1/mean2)
#         for Galaxy in Gamma:
#             gamma_real.append(Galaxy['gamma1'])
#     z = np.polyfit(gamma_real, gamma_cal, 1)
#     m = z[0]
#     b = z[1]
#     cs_uncorr.append(b)
# plt.plot(r_halfs, cs, 'ro', label = 'KSB')
# plt.plot(r_halfs, cs_uncorr, color = 'orange', marker = 'o', linestyle = '', label = 'uncorrected')
# plt.xlabel('$r_h$ [arcsec]')
# plt.ylabel('$c_1$')
# plt.legend()



#b = 1#0.66

# bsm = np.linspace(1.1,1.4,3)
# gamma_real = np.linspace(-0.1, 0.1, 20)
# bs_err = []
# bs_real = []
# e_psfs = np.linspace(0,0.1,6)
# for j in range(6):
    # cs = []
    # cs_err = []
#     for b in bsm:
#         gamma_cal = []
#         Errors = []
#         for i in range(20):
#             path = config.workpath('Run' + str(j+1) + '/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
#             table = Table.read(path)
#             e1 = table['e1_cal']
#             e_corr = table['anisotropy_corr']
#             P_g11 = table['Pg_11']
#             Flags = table['Flag']
#             Is_good = Flags == 0
#             mean1 = np.mean(e1[Is_good]-b*e_corr[Is_good])
#             mean2 = np.mean(P_g11[Is_good])
#             gamma_cal.append(mean1/mean2)
#             Errors.append(Bootstrap(table, b, True))
#         popt, pcov = curve_fit(linear, gamma_real, gamma_cal, sigma = Errors, absolute_sigma = True)
#         m = popt[0]
#         bias = popt[1]
#         cs.append(bias) 
#         cs_err.append(pcov[1,1])
#     popt, pcov = curve_fit(linear, bsm, cs, sigma = cs_err, absolute_sigma = True)
#     m2 = popt[0]
#     b2 = popt[1]
#     bs_real.append(-1*b2/m2)
#     bs_err.append(np.sqrt((1/m2*pcov[1,1])**2 + (b2/m2**2*pcov[0,0])**2))
#     print('boost factor determined with' + str(-1*b2/m2) + ' and ' + str(np.sqrt((1/m2*pcov[1,1])**2 + (b2/m2**2*pcov[0,0])**2)))
    
# plt.errorbar(e_psfs, bs_real, yerr = bs_err, ecolor = 'red', fmt = 'ro')
# plt.xlabel('$e_1^{PSF}$')
# plt.ylabel('$b^{sm}_1$')
# plt.savefig('Plots2/bsm_e_psf2.pdf')

# gamma_real = np.linspace(-0.1,0.1,20)
# gamma_cal = []
# for i in range(20):
#     path = config.workpath('Test_' + str(i+1) +'/Measured_galsim.fits')
#     table = Table.read(path)
#     g1 = table['g1_cal_galsim']
#     Is_good = g1 > -9
#     gamma_cal.append(np.mean(g1[Is_good]))
# print(Is_good)


# b = 1
# bss = [1.5539611125802553, 1.2647693321461664, 1.2632639648341009, 1.2317227364998693, 1.2262127036091375, 1.2257287304708997]
# cs_corr = []
# cs_corr2 = []
# cs_uncorr = []
# e_psfs = np.linspace(0,0.1,6)
# gamma_real = np.linspace(-0.1,0.1,20)
# for j in range(6):
#     gamma_cal = []
#     Errors = []
#     for i in range(20):
#         path = config.workpath('Run' + str(j+1) + '/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
#         table = Table.read(path)
#         e1 = table['e1_cal']
#         e_corr = table['anisotropy_corr']
#         P_g11 = table['Pg_11']
#         mean1 = np.mean(e1-b*e_corr)
#         mean2 = np.mean(P_g11)
#         gamma_cal.append(mean1/mean2 - gamma_real[i])
#         Errors.append(Bootstrap(table, b, False))
#     popt, pcov = curve_fit(linear, gamma_real, gamma_cal, sigma = Errors, absolute_sigma = True)
#     cs_corr.append(popt[1])
#     print('corrected bias finished')
# for j in range(6):path = config.workpath('Test5')

#     gamma_cal = []
#     Errors = []
#     for i in range(20):
#         path = config.workpath('Run' + str(j+1) + '/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
#         table = Table.read(path)
#         e1 = table['e1_cal']
#         e_corr = table['anisotropy_corr']
#         P_g11 = table['Pg_11']
#         mean1 = np.mean(e1- bss[j] *e_corr)
#         mean2 = np.mean(P_g11)
#         gamma_cal.append(mean1/mean2 - gamma_real[i])
#         Errors.append(Bootstrap(table, bss[j]))
#     popt, pcov = curve_fit(linear, gamma_real, gamma_cal, sigma = Errors, absolute_sigma = True)
#     cs_corr2.append(popt[1])
#     print('corrected bias finished with ' + str(popt[1]))
# for j in range(6):
#     gamma_cal = []
#     Errors = []
#     for i in range(20):
#         path = config.workpath('Run' + str(j+1) + '/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
#         table = Table.read(path)
#         e1 = table['e1_cal']
#         P_g11 = table['Pg_11']
#         mean1 = np.mean(e1)
#         mean2 = np.mean(P_g11)
#         gamma_cal.append(mean1/mean2 - gamma_real[i])
#         Errors.append(Bootstrap(table, 0, False))
#     popt, pcov = curve_fit(linear, gamma_real, gamma_cal, sigma = Errors, absolute_sigma = True)
#     cs_uncorr.append(popt[1])
# plt.plot(e_psfs, cs_corr, 'ro', label = 'KSB')
# plt.plot(e_psfs, cs_corr2, 'go', label = 'KSB with boost factor')
# plt.plot(e_psfs, cs_uncorr,color = 'orange', marker = 'o', linestyle = '', label = 'uncorrected')
# plt.xlabel('$e_1^{PSF}$')
# plt.ylabel('$c_1$')
# plt.legend()
        
# path = config.workpath('Test_1/Measured_ksb.fits')
# table = Table.read(path)

# Pg11 = table['Pg_11']
# Pg22 = table['Pg_22']
# plt.plot(Pg11, Pg22, '.')
# plt.xlabel('$P^\gamma_{11}$')
# plt.ylabel('$P^\gamma_{22}$')

# path2 = config.workpath('Test_1/Gamma.fits')
# Gamma = Table.read(path2)
# e1_cal = table['e1_cal']
# e1 = table['e1']
# e_corr = table['anisotropy_corr']
# P_g = table['Pg_11']   
# Rot = table['rotation'] == 1
# Pg_good1 = (P_g > 0.185)
# Pg_good2 = (P_g < -0.185)
# mag = table['mag']
# data = []
# print(P_g, np.mean(np.abs(P_g)))
# mag_range = mag > 21
# mag_range2 = mag < 22
# Is_good = np.logical_and(mag_range, mag_range2)
# r_half = table['r_half'] 
# n_ser = table['n']
# # print(np.mean(P_g[Is_good]))
# # Not_good = np.invert(Is_good) 
# mags = [20.5,21.5,22.5,23.5,24.5]
# r_halfs = np.linspace(0,1.1, 12)
# ns = np.linspace(0,6, 10)

# # for i in mags:
# #     mag_range = mag > i - 0.5
# #     mag_range2 = mag < i + 0.5
# #     Is_good = np.logical_and(mag_range, mag_range2)
# #     data.append(np.mean(P_g[Is_good]))
# for i in r_halfs:
#     r_range = r_half > i 
#     r_range2 = r_half < i + 1.1/12
#     Is_good1 = np.logical_and(r_range, r_range2)
#     test = P_g > 0.03
#     Is_good = np.logical_and(Is_good1, test)
#     data.append(np.mean(np.abs(P_g[Is_good])))
# # for i in ns:
# #     nss = n_ser > i 
# #     nss2 = n_ser < i + 6/10
# #     Is_good = np.logical_and(nss, nss2)
# #     data.append(np.mean(P_g[Is_good]))
# plt.scatter(r_halfs, data)
# plt.xlabel('sersic index bins')
# plt.ylabel('pre-seeing shear polarisability')

    
# path = config.workpath('Test_1/Measured_ksb.fits')
# table = Table.read(path)
# Pg = table['Pg_11']
# e1 = table['e1_cal']
# an_cor = table['anisotropy_corr']
# plt.plot(e1, an_cor, '.')




# for i in range(1,2):
#     path = config.workpath('Run6/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
#     table = Table.read(path)
#     n_sersic = table['n']
#     mag = table['mag']
#     rhalf = table['r_half']
#     R = table['radius_est']*0.02*np.sqrt(2*np.log(2))
#     Aper_sum = table['aperture_sum'] 
#     Mag_estimate = 24.6 - 2.5 * np.log10(3.1/(3*565)* Aper_sum)
#     # plt.plot(mag, Mag_estimate, '.')
#     plt.scatter(rhalf, R, c = n_sersic, marker = '.', cmap = 'viridis')
#     plt.xlabel('GEMS half light radius in arcsec')
#     plt.ylabel('half light radius estimate in arcsec')
#     plt.colorbar()
#     plt.savefig('Plots2/r-half_r_half.pdf')
#     # plt.xlabel('GEMS magnitude')
#     # plt.ylabel('magnite estimate')


# gamma_real = []
# gamma_cal = []
# Errors = []
# b = 1.25
# for i in range(20):
#     path = config.workpath('Run2/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
#     table = Table.read(path)
#     path2 = config.workpath('Run6/PSF_es_' + str(i+1) +'/Gamma.fits')
#     Gamma = Table.read(path2)
#     e1 = table['e1_cal']
#     e_corr = table['anisotropy_corr']
#     P_g11 = table['Pg_11']
#     R = table['radius_est']*0.02 #in arcsec
#     R_half = table['r_half']
#     xc = table['x_centroid']
#     yc = table['y_centroid']
#     Aper_sum = table['aperture_sum'] 
#     gamma_errs = []
#     for i in range(100):
#         rng = np.random.default_rng()
#         numb_random = rng.choice(range(table.meta['NY_TILES'] * table.meta['NX_TILES']), size = (table.meta['NY_TILES'] * table.meta['NX_TILES']))
#         e1_err1 = table['e1_cal'][2*numb_random-2]
#         e1_err2 = table['e1_cal'][2*numb_random-1]
#         e1_err = np.concatenate((e1_err1, e1_err2))
#         anis_err1 = table['anisotropy_corr'][2*numb_random-2]
#         anis_err2 = table['anisotropy_corr'][2*numb_random-1]
#         anis_err = np.concatenate((anis_err1, anis_err2))
#         Pg_err2 = table['Pg_11'][2*numb_random-2]
#         Pg_err1 = table['Pg_11'][2*numb_random-1]
#         Pg_err = np.concatenate((Pg_err1, Pg_err2))
#         Flags1 = table['Flag'][2*numb_random-2]
#         Flags2 = table['Flag'][2*numb_random-1]  
#         Flags = np.concatenate((Flags1, Flags2))
#         Is_good = Flags == 0
#         pol_err = np.mean(e1_err[Is_good] - b* anis_err[Is_good])
#         Pg_err = np.mean(Pg_err[Is_good])
#         gamma_errs.append(pol_err/Pg_err)
#     Errors.append(np.sqrt(np.mean((gamma_errs-np.mean(gamma_errs))**2)))
#     Flag = table['Flag']
#     Is_good = Flag == 0
#     polarisation = np.mean(e1-b*e_corr)
#     Pg = np.mean(P_g11)
#     #mean3 = np.mean(e1)
#     gamma_cal.append(polarisation/Pg)
#     #print(mean3-mean1, mean2)
#     for Galaxy in Gamma:
#         gamma_real.append(Galaxy['gamma1'])
# gamma_dif = [] 
# for i in range(20):
#     gamma_dif.append(gamma_cal[i] - gamma_real[i])


# popt, pcov = curve_fit(linear, gamma_real, gamma_dif, sigma = Errors, absolute_sigma = True)

# g = np.linspace(-0.1, 0.1, 250)
# plt.errorbar(gamma_real, gamma_dif, yerr = Errors, ecolor = 'red', fmt = 'r.')

# z = np.polyfit(gamma_real, gamma_dif, 1)
# m = z[0]
# b = z[1]
# plt.plot(g, linear(g, m, b))
# plt.xlabel('$g_{input}$')
# plt.ylabel('$g_{output} - g_{input}$')
# plt.title('$\mu$ = ' + str(round((m),6)) + '$\pm$ ' + str(round(pcov[0,0],6)) + '  $c$ = ' + str(round(b,6)) + '$\pm$ ' + str(round(pcov[1,1],8)))
# # plt.savefig('Plots2/Bootstrapping_Gamma.pdf')

# b = 1
# gamma_cal = []
# gamma_real = []
# for i in range(20):
#     path = config.workpath('Run6/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
#     table = Table.read(path)
#     path2 = config.workpath('Run6/PSF_es_' + str(i+1) +'/Gamma.fits')
#     Gamma = Table.read(path2)
#     e1 = table['e1_cal']
#     e_corr = table['anisotropy_corr']
#     P_g11 = table['Pg_11']
#     Aper_sum = table['aperture_sum'] 
#     Is_good = Aper_sum > 600
#     Is_good = Is_good.T
#     polarisation = np.mean(e1[Is_good[0]]-b*e_corr[Is_good[0]])
#     Pg = np.mean(P_g11[Is_good[0]])
#     #mean3 = np.mean(e1)
#     gamma_cal.append(polarisation/Pg)
#     #print(mean3-mean1, mean2)
#     for Galaxy in Gamma:
#         gamma_real.append(Galaxy['gamma1'])
# path3 = config.workpath('Run6/Errors.fits')
# Errors = Table.read(path3)
end = time.time()
total_time = (end - start)/(60)  #run time in hours
print('The system took ', total_time ,' minutes to execute the function')    
    
