
import galsim
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import tab
import image
import logging
import time
import ksb
import config
from astropy.table import Table

start = time.time()

def generate_sim_trainingSet(path, case):
    mydir =config.workpath(path)
    tab.training_set_tab(5, 10, case, 10, 64, 64, mydir)
    cases = np.arange(case)
    #gal_image = []
    final = []
    for i in range(case):
        params = [mydir + '/Input_data_' + str(i) + '.fits', mydir, cases[i]]
        final.append(params)
    with Pool() as pool:
        pool.starmap(image.generate_realisations, final)
    return None

#generate_sim_trainingSet('Test', 4)    


def generate_simulation(only_one_table = False, path_table = 'Test/table.fits', dirname='test', gamma1 = 0, gamma2 = 0, psf_pol = 0):   
    mydir = config.workpath(dirname)
    # mylogpath = os.path.join(mydir, "generate_simulation.log")
    # log_format = '%(asctime)s %(filename)s: %(message)s'
    # logging.basicConfig(filename=mylogpath, format=log_format, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    # logging.info('started simulation')
    if only_one_table == True:
        #logging.info('Simulation uses only one randomly drawn table of galaxy parameters for the generated grid')
        t = Table.read(path_table)
        tab.generate_gamma_tab(mydir, gamma1, gamma2, psf_pol)
        file_name = os.path.join(mydir, '/Input_data.fits')
        t.write(file_name, overwrite = True)
        image.generate_image(mydir + '/Input_data.fits', mydir)
    else:
        #logging.info('Simulation generates a new randomly drawn table of galaxy parameters for the grid')
        os.makedirs(mydir, exist_ok=True)
        tab.generate_gamma_tab(mydir, gamma1, gamma2, psf_pol)
        tab.generate_table(100,100,64,64, mydir)
        image.generate_image(mydir + '/Input_data.fits', mydir)
    return None

#ksb.calculate_ksb(mydir) #getrennt
#generate_simulation(True, 'Test/table.fits', 'output1')
#image.generate_psf_image('Test/table.fits')

def simulate_Grids(N, name, psf_pol):
    runname = name
    final = []
    Gammas = np.linspace(-0.1, 0.1, N)
    gamma1 = 0
    for i in range(N):
        gamma1 = Gammas[i]
        params = [False,'Test/Input_data.fits' ,runname + "_" + str(i+1), gamma1, 0, psf_pol]
        final.append(params)
    with Pool() as pool:
        pool.starmap(generate_simulation, final)



def simulate_Grids_psf(N_gammas, N_psf_pols, name1, name2, psf_pol):
    final = []
    Gammas = np.linspace(-0.1,0.1, N_gammas)
    psf_pols = np.linspace(0, psf_pol, N_psf_pols)
    gamma1 = 0
    psf_pol_dum = 0
    for i in range(N_psf_pols):
        for j in range(N_gammas):
            gamma1 = Gammas[j]
            psf_pol_dum = psf_pols[i]
            params = [False, '', name1 + str(i+1) + name2 + '_' + str(j+1), gamma1, 0, psf_pol_dum]
            final.append(params)
    with Pool() as pool:
        pool.starmap(generate_simulation, final)

def calculate_shear_psf(N_gammas, N_psf_pols, name1, name2):
    final = []
    for i in range(N_psf_pols):
        for j in range(N_gammas):
            params = [config.workpath(name1 + str(i+1) + name2 + '_' + str(j+1))]
            final.append(params)
    with Pool() as pool:
        pool.starmap(ksb.calculate_ksb, final)


def calculate_shear_galsim(N):
    runname = 'Test'
    final = []
    for i in range(N):
        params = [config.workpath(runname + '_' + str(i+1))]
        final.append(params)
    with Pool() as pool:
        pool.starmap(ksb.calculate_ksb_galsim, final)


def calculate_shear(N, runname):
    runname = runname
    final = []
    for i in range(N):
        params = [config.workpath(runname + '_' + str(i+1))]
        final.append(params)
    with Pool() as pool:
        pool.starmap(ksb.calculate_ksb, final)

# simulate_Grids(20, 'Run6/PSF_es', 0.1)
# calculate_shear(20, 'Run6/PSF_es')  

end = time.time()
total_time = (end - start)/(60*60)  #run time in hours
#print('The system took ', total_time ,' hours to execute the function')
