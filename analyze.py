import config
from astropy.table import Table
import galsim
import matplotlib.pyplot as plt

gamma_real = []
gamma_cal = []
for i in range(20):
    path = config.workpath('Test_' + str(i+1) +'/Measured.fits')
    table = Table.read(path)
    path2 = config.workpath('Test_' + str(i+1) +'/Gamma.fits')
    Gamma = Table.read(path2)
    dum1 = 0
    dum2 = 0
    for Galaxy in table:
        shear = galsim.Shear(e1 = Galaxy['e1_cal'], e2 = Galaxy['e2_cal'])
        gamma1 = shear._g.real
        dum1 = dum1 + gamma1
        gamma2 = shear._g.imag
        dum2 = dum2 + Gamma['gamma1']
    dum1 = dum1/len(table) #np.mean
    gamma_cal.append(dum1)
    dum2 = dum2/len(table)
    gamma_real.append(dum2)


#anisotropy correction
# print(gamma_cal)
# print(gamma_real)
plt.plot(gamma_real, gamma_cal, 'o')
    
    
    
