
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
import analyze
from astropy.table import Table, vstack

# start = time.time()

def generate_sim_trainingSet(path, case):
    mydir =config.workpath(path)
    os.makedirs(mydir, exist_ok=True)
    tab.training_set_tab(3, 20, case, 200, 64, 64, 350, 350, mydir)
    cases = np.arange(case)
    #gal_image = []
    final = []
    for i in range(case):
        params = [mydir, cases[i]]
        final.append(params)
    with Pool() as pool:
        pool.starmap(image.generate_realisations, final)
    table = Table(names = ['mag', 'n', 'r_half', 'e1', 'e2', 'gamma1'], dtype = ['f4', 'f4', 'f4', 'f4', 'f4', 'f4'])
    for i in range(case):
        table1 = Table.read(mydir + '/Input_data_'+ str(i) + '.fits')
        table = vstack([table,table1]) 
    if not os.path.isdir(mydir):
        os.mkdir(mydir)
    file_name = os.path.join(mydir, 'Input_data.fits')
    table.write( file_name , overwrite=True) 
    return None
# generate_sim_trainingSet('Test2', 1)

def ksb_training(path, case):
    mydir = config.workpath(path)
    cases = np.arange(case)
    final = []
    for i in range(case):
        params = [mydir, cases[i]]
        final.append(params)
    with Pool() as pool:
        pool.starmap(ksb.calculate_ksb_training, final)
    table = Table(names = ['mag', 'n', 'r_half', 'e1', 'e2', 'gamma1', 'b_sm'], dtype = ['f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'])
    for i in range(case):
        table1 = Table.read(mydir + '/Measured_ksb_'+ str(i) + '.fits')
        table = vstack([table,table1]) 
    if not os.path.isdir(mydir):
        os.mkdir(mydir)
    file_name = os.path.join(mydir, 'Measured_ksb.fits')
    table.write( file_name , overwrite=True)     
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

# end = time.time()
# total_time = (end - start)/(60*60)  #run time in hours
#print('The system took ', total_time ,' hours to execute the function')
