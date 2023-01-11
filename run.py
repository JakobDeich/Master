import simulation
import config
import numpy as np
import time
import os
import ksb
import analyze
import tab
import image
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

#simulation.simulate_Grids_psf(20, 6, 'Run', '/PSF_es', 0.1)
# simulation.calculate_shear_psf(20, 6, 'Run', '/PSF_es')

if __name__ == "__main__":
    # simulation.generate_sim_trainingSet('Test5', 2)
    simulation.ksb_training('Test4', 200)
    # simulation.ksb_training('Test', 200)
    # simulation.ksb_training('Test2', 200)

# path = config.workpath('Example')
# tab.tab_realisation(1, 1, 1, 1, 64, 64, 350, 350, 22, 1, 1.2, 0.03, 668, path)
# image.generate_realisations(path + '/Input_data_668.fits', path, 668)

# for i in range(100):
#     ksb.calculate_ksb_training('Test', i)
#     analyze.determine_boost('Test', i)


end = time.time()
total_time = (end - start)/(60*60*24)  #run time in days
print('The system took ', total_time ,' days to execute the function')
