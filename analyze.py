import config
from astropy.table import Table
import galsim
import matplotlib.pyplot as plt
import statistics
import numpy as np
from matplotlib import cm

def linear(x, m, b):
    return m*x + b

e1_s = []
gamma_cal = []
#calculate P^gamma to get shear estimate


# for i in range(20):
#     path = config.workpath('Test_' + str(i+1) +'/Measured_ksb.fits')
#     table = Table.read(path)
#     e1 = table['e1_cal']
#     e1_corr = table['anisotropy_corr']
#     Pg11 = table['Pg_11']
#     mean1 = np.mean(e1)
#     mean2 = np.mean(e1_corr)
#     mean3 = np.mean(Pg11)
#     e1_s.append(mean1-mean2)
#     gamma_cal.append((mean1-mean2)/mean3)
# gamma_real = np.linspace(-0.1, 0.1, 20)
# z = np.polyfit(gamma_real, e1_s, 1)
# P_g = z[0]
# print(P_g)
# #calculate shear estimate
# gamma_cal = e1_s/P_g

gamma_real = []  
b = 0.0 #0.66
for i in range(20):
    path = config.workpath('Test_' + str(i+1) +'/Measured_ksb.fits')
    table = Table.read(path)
    path2 = config.workpath('Test_' + str(i+1) +'/Gamma.fits')
    Gamma = Table.read(path2)
    mag = table['mag']
    mag_range = mag > 21
    mag_range2 = mag < 22
    Is_good = np.logical_and(mag_range, mag_range2)
    e1 = table['e1_cal']
    r_half = table['r_half']
    e_corr = table['anisotropy_corr']
    P_g11 = table['Pg_11']
    #print(e_corr)
    #print(P_g11)
    mean1 = np.mean(e1 - (b* e_corr))
    mean2 = np.mean(P_g11)
    #mean3 = np.mean(e1)
    gamma_cal.append(mean1/mean2)
    #print(mean3-mean1, mean2)
    for Galaxy in Gamma:
        gamma_real.append(Galaxy['gamma1'])
        
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

    
    
# plt.plot(P_g, e1_cal-e_corr, 'b.', label ='data')
# plt.plot(np.mean(P_g), np.mean(e1_cal-e_corr), 'r.', label = 'mean')
# plt.xlabel('pre-seeing shear polarisability')
# plt.ylabel('corrected polarisation')
# plt.legend()
# plt.plot(r_half[Not_good],P_g[Not_good], 'b.', label = 'unsuccesfull')
# plt.scatter(r_half, n_ser, c = P_g, cmap = 'rainbow', s = 0.3)
# # fig.colorbar(psm, ax=ax)
# plt.ylabel('sersic index')
# plt.xlabel('r_half')
# cbar = plt.colorbar()
# cbar.set_label('pre-seeing shear polarizability')

# plt.hist(P_g, bins= 'auto')
# plt.xlabel('pre-seeing shear polarizability')
# plt.ylabel('counts')

# gamma_real = []
# for i in range(1):
#     path = config.workpath('Test_' + str(i+20) +'/Measured_galsim.fits')
#     table = Table.read(path)
#     path2 = config.workpath('Test_' + str(i+20) +'/Gamma.fits')
#     Gamma = Table.read(path2)
#     dum1 = []
#     dum2 = []
#     for Galaxy in table:
#         if Galaxy['g1_cal_galsim'] > -10:
#             # shear = galsim.Shear(e1 = Galaxy['e1_cal_galsim'], e2 = Galaxy['e2_cal_galsim'])
#             gamma1 = Galaxy['g1_cal_galsim']
#             gamma2 = Galaxy['g2_cal_galsim']
#             #print(gamma1)
#             e1 = Galaxy['e1_galsim']
#             dum1.append(e1)
#             dum2.append(gamma1)
        
#     # for Galaxy in Gamma:
#     #     # gamma2 = shear._g.imag
#     #     dum2.append(Galaxy['gamma1'])
#     mean1 = statistics.mean(dum1)
#     gamma_cal.append(mean1)
#     mean2 = statistics.mean(dum2)
#     gamma_real.append(mean2)
#     print(mean1, mean2)


g = np.linspace(-0.1, 0.1, 250)
plt.plot(gamma_real, gamma_cal, 'o')

z = np.polyfit(gamma_real, gamma_cal, 1)
m = z[0]
b = z[1]
print(z)
plt.plot(g, linear(g, m, b))
plt.xlabel('shear $g\_{input}$')
plt.ylabel('shear $g\_{output}$')
plt.title('$\mu$ = ' + str(m - 1) + '  $c$ = ' + str(b))



    
    
