import simulation
import config
import numpy as np
import time
import os
from multiprocessing import Pool

start = time.time()

def psf_pol(psf_es, number):
    name = 'Run' + str(number) + '/PSF_es'
    simulation.simulate_Grids(20, name , psf_es)
    simulation.calculate_shear(20, name)



def psf_pol_run(N = 6, psf_pol_max = 0.1):
    psf_es = np.linspace(0,psf_pol_max,N)
    number = list(range(1,N+1))
    for i in range(N):
        psf_pol(psf_es[i], number[i])
#psf_pol_run(6,0.1)

simulation.simulate_Grids_psf(20, 6, 'Run', '/PSF_es', 0.1)
simulation.calculate_shear_psf(20, 6, 'Run', '/PSF_es')

end = time.time()
total_time = (end - start)/(60*60*24)  #run time in days
print('The system took ', total_time ,' days to execute the function')
