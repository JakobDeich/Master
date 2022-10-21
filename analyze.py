import config
from astropy.table import Table
import galsim
import matplotlib.pyplot as plt
import statistics
import numpy as np
from matplotlib import cm
from scipy.optimize import curve_fit
from astropy.table import Table
import os
from astropy.table import Table, Column
import time

start = time.time()

def linear(x, m, b):
    return m*x + b

def gal_mag(flux):
    return  24.6 - 2.5 * np.log10(3.1/(3*565)* flux)


# b = 1
# gamma_cal = []
# gamma_real = np.linspace(-0.1,0.1,20)
# bsm = np.linspace(1,1.4,4)
# cs = []
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
#     # plt.scatter(gamma_real, gamma_cal, label = '$b^{sm}_1$ = ' + str(b) + ' , $\mu_1$ = ' + str(round(popt[0], 4)) + ' , $c_1$ = ' + str(round(popt[1], 4)))
#     # plt.plot(gamma_real, linear(gamma_real, popt[0], popt[1]))
#     cs.append(popt[1]) 
# popt, pcov = curve_fit(linear, bsm, cs)
# plt.scatter(bsm, cs)
# plt.plot(bsm, linear(bsm, popt[0], popt[1]))
# m2 = popt[0]
# b2 = popt[1]
# plt.scatter(-1*b2/m2, 0, s = 80, color = 'r', marker = 'D',label = 'target boost factor')    
# plt.xlabel('boost factor $b^{sm}_1$')
# plt.ylabel('additive bias $c_1$')
# plt.legend()
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


def determine_boost(path, case):
    print('determining boost factor for case ' + str(case))
    mydir = config.workpath(path)
    table = Table.read(mydir + '/Measured_ksb_' + str(case) + '.fits')
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

    
# determine_boost('Test2',3)
# mydir = config.workpath('Test')
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
# for j in range(6):
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
    
