import simulation
import config
import numpy as np
import time
import os

start = time.time()

for i in range(6):
    name = 'Run' + str(i+1) + '/PSF_es'
    psf_es = np.linspace(0,0.1,6)
    simulation.simulate_Grids(20, name , psf_es[i])
    simulation.calculate_shear(20, name)


end = time.time()
total_time = (end - start)/(60*60*24)  #run time in days
print('The system took ', total_time ,' days to execute the function')

