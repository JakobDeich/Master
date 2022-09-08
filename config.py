import os
from astropy.table import Table
import numpy as np

#workdir = "/vol/aibn1053/data1/jdeich/work_master"
workdir = '/vol/euclidraid4/data/jdeich'


 
def workpath(relpath):
    return os.path.join(workdir, relpath)

# Gamma = Table.read('Test/Gamma.fits')
# gamma1 = Gamma['gamma1']
# print(gamma1)


# gamma_real = []
# for i in range(20):
#     gamma_real.append((i*0.2)/20 - 0.1)
# g = np.linspace(-0.1,0.1,20)
# print(gamma_real, g, len(gamma_real))