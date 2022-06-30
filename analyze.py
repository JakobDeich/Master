import config
from astropy.table import Table
import galsim
import matplotlib.pyplot as plt
import statistics
import numpy as np

def linear(x, m, b):
    return m*x + b

e1 = []
gamma_real = []
gamma_cal = []
#calculate P^gamma to get shear estimate
# for i in range(20):
#     path = config.workpath('Test_' + str(i+1) +'/Measured_ksb.fits')
#     table = Table.read(path)
#     path2 = config.workpath('Test_' + str(i+1) +'/Gamma.fits')
#     Gamma = Table.read(path2)
#     dum1 = []
#     dum2 = []
#     for Galaxy in table:
#         if Galaxy['rotation'] == 0 and np.sqrt(Galaxy['e1_cal']**2 + Galaxy['e2_cal']**2) < 1:
#             dum1.append(Galaxy['e1_cal'])
#     for Galaxy in Gamma:
#         dum2.append(Galaxy['gamma1'])
#     mean1 = statistics.mean(dum1)
#     e1.append(mean1)
#     mean2 = statistics.mean(dum2)
#     gamma_real.append(mean2)
# #plt.plot(gamma_real, e1, 'o')  
# g = np.linspace(-0.1, 0.1, 250)

# z = np.polyfit(gamma_real, e1, 1)
# P_g = z[0]
# print(P_g)
# #calculate shear estimate
# for i in range(20):
#     path = config.workpath('Test_' + str(i+1) +'/Measured_ksb.fits')
#     table = Table.read(path)
#     dum1 = []
#     for Galaxy in table:
#         if Galaxy['rotation'] == 0 and np.sqrt(Galaxy['e1_cal']**2 + Galaxy['e2_cal']**2) < 1:
#             dum1.append(Galaxy['e1_cal'])
#         # print(Galaxy['e1_cal'])
#     mean1 = statistics.mean(dum1)
#     gamma_cal.append(mean1/P_g)

for i in range(20):
    path = config.workpath('Test_' + str(i+1) +'/Measured_ksb.fits')
    table = Table.read(path)
    path2 = config.workpath('Test_' + str(i+1) +'/Gamma.fits')
    Gamma = Table.read(path2)
    dum1 = []
    dum2 = []
    for Galaxy in table:
        if np.sqrt(Galaxy['e1_cal']**2 + Galaxy['e2_cal']**2) < 1:
            gamma1 = Galaxy['g1_cal']
            gamma2 = Galaxy['g2_cal']
            #print(gamma1)
            dum1.append(gamma1)
        
    for Galaxy in Gamma:
        # gamma2 = shear._g.imag
        dum2.append(Galaxy['gamma1'])
    mean1 = statistics.mean(dum1)
    gamma_cal.append(mean1)
    mean2 = statistics.mean(dum2)
    gamma_real.append(mean2)


# for i in range(20):
#     path = config.workpath('Test_' + str(i+1) +'/Measured_galsim.fits')
#     table = Table.read(path)
#     path2 = config.workpath('Test_' + str(i+1) +'/Gamma.fits')
#     Gamma = Table.read(path2)
#     dum1 = []
#     dum2 = []
#     for Galaxy in table:
#         if Galaxy['g1_cal_galsim'] > -10 and Galaxy['rotation'] == 0:
#             # shear = galsim.Shear(e1 = Galaxy['e1_cal_galsim'], e2 = Galaxy['e2_cal_galsim'])
#             gamma1 = Galaxy['g1_cal_galsim']
#             gamma2 = Galaxy['g2_cal_galsim']
#             #print(gamma1)
#             dum1.append(gamma1)
        
#     for Galaxy in Gamma:
#         # gamma2 = shear._g.imag
#         dum2.append(Galaxy['gamma1'])
#     mean1 = statistics.mean(dum1)
#     gamma_cal.append(mean1)
#     mean2 = statistics.mean(dum2)
#     gamma_real.append(mean2)


g = np.linspace(-0.1, 0.1, 250)
plt.plot(gamma_real, gamma_cal, 'o')

z = np.polyfit(gamma_real, gamma_cal, 1)
m = z[0]
b = z[1]
print(z)
plt.plot(g, linear(g, m, b))
plt.xlabel('shear $g\_{input}$')
plt.ylabel('shear $g\_{output}$')
plt.title('$\mu$ = ' + str(m) + '  $c$ = ' + str(b))

    

    
    
