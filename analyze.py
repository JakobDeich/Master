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
import time

start = time.time()

def linear(x, m, b):
    return m*x + b

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

def boostFactorDep(Var, Var_name, N):
    path = config.workpath('Run6/PSF_es_1/Measured_ksb.fits')
    table = Table.read(path)
    Vars1 = table[Var]
    min_var = min(Vars1)
    max_var = max(Vars1)
    step = (max_var-min_var)/N
    Var_plot = []
    bsm = np.linspace(1.1,1.4,3)
    gamma_real = np.linspace(-0.1, 0.1, 20)
    bs_err = []
    bs_real = []
    for j in range(N):
        cs = []
        cs_err = []
        dum = min_var + j*step + 0.5*step
        dum.append(Var_plot)
        for b in bsm:
            gamma_cal = []
            Errors = []
            for i in range(20):
                path = config.workpath('Run6/PSF_es_' + str(i+1) +'/Measured_ksb.fits')
                table = Table.read(path)
                e1 = table['e1_cal']
                e_corr = table['anisotropy_corr']
                P_g11 = table['Pg_11']
                Flags = table['Flag']
                Vars = table[Var]
                Is_good1 = Flags == 0
                Is_good2 = Vars >= (min_var + j*step)
                Is_good3 = Vars < (min_var + (j+1)*step)
                Is_good4 = np.logical_and(Is_good2, Is_good3)
                Is_good = np.logical_and(Is_good1, Is_good4)
                mean1 = np.mean(e1[Is_good]-b*e_corr[Is_good])
                mean2 = np.mean(P_g11[Is_good])
                gamma_cal.append(mean1/mean2)
                Errors.append(Bootstrap(table, b, True))
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
        print('boost factor determined with' + str(-1*b2/m2) + ' and ' + str(np.sqrt(np.sqrt((1/m2*pcov[1,1])**2 + (b2/m2**2*pcov[0,0])**2))))
    plt.errorbar(Var_plot, bs_real, yerr = bs_err, ecolor = 'red', fmt = 'ro')
    plt.xlabel(Var_name)
    plt.ylabel('$b^{sm}_1$')
    plt.savefig('boost_' + Var_name + '.pdf')
    return 0

boostFactorDep('sigma_mom', 'sigma moments', 6)
boostFactorDep('rho4_mom', 'rho4', 6)
boostFactorDep('aperture_sum', 'aperture sum', 6)

e1_s = []

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
    
