import simulation
import config
import numpy as np
import time
import os
import ksb
import analyze
import tab
import galsim
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
    simulation.generate_sim_trainingSet('Test', 100)
    simulation.ksb_training('Test', 100)
    final = []
    # final1 = []
    # path = config.workpath('Test')
    cases = [0,20,30,40,50, 60, 70]
    trys =np.arange(15)
    # for j in cases:
    #     for i in trys:
    #         try:
    #             image_file = path + '/Grid_case' + str(j) +'_' + str(i) + '.fits'
    #             gal_image = galsim.fits.read(image_file)
    #         except:
    #             params = [config.workpath('Test5'), j, i] 
    #             final1.append(params)
    for j in cases:
        for i in trys:
            params = [config.workpath('Test'), j, i]
            final.append(params)
    with Pool() as pool:
        pool.starmap(image.generate_realisations_trys, final)
    with Pool() as pool:
        pool.starmap(ksb.calculate_ksb_training_trys, final)


# path = config.workpath('Example')
# tab.tab_realisation(1, 1, 1, 1, 64, 64, 350, 350, 22, 1, 1.2, 0.03, 668, path)
# image.generate_realisations(path + '/Input_data_668.fits', path, 668)

# for i in range(100):
#     ksb.calculate_ksb_training('Test', i)
#     analyze.determine_boost('Test', i)


end = time.time()
total_time = (end - start)/(60*60*24)  #run time in days
print('The system took ', total_time ,' days to execute the function')
