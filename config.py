import os
from astropy.table import Table
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import threading
#workdir = "/vol/aibn1053/data1/jdeich/work_master"
workdir = '/vol/euclidraid4/data/jdeich'
#workdir = ''
def workpath(relpath):
    return os.path.join(workdir, relpath)


# Gamma = Table.read('Test/Gamma.fits')
# gamma1 = Gamma['gamma1']
# print(gamma1)
# print(plt.get_backend())
# a = np.ones((2,3))*0.00000000001
# b = np.ones((2,3))*0.00000000003
# print(np.subtract(a,b))
#print('hello')
#time.sleep(5)
#print(threading.active_count())
#print('bye')
# gamma_real = []
# for i in range(20):
#     gamma_real.append((i*0.2)/20 - 0.1)
# g = np.linspace(-0.1,0.1,20)
# print(gamma_real, g, len(gamma_real))

# plt.plot([0,1,2,3], [2,2,2,2])
# plt.show()

# def s(r,n):
#     return -(1.999*n-0.327)*(r**(1/n)-1)

# x = np.linspace(0,5, 600)
# ns = [0.5,0.75,1,2,3,4]
# for i in ns:
#     plt.plot(x, s(x,i), label = 'n = ' + str(i))
# plt.xlabel('radius in units of $R_e$')
# plt.ylabel('log surface brightness in units of $I_e$')
# plt.legend()
# plt.rcParams['font.size'] = 22

