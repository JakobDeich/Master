
from astropy.table import Table
from astropy.io import ascii
import numpy as np
import galsim
import random
import os
import logging
import time
import config
from multiprocessing import Pool
#import numpy.random.Generator

def trunc_rayleigh(sigma, max_val, case):
    np.random.seed(case)
    assert max_val > sigma
    tmp = max_val + 1
    while tmp > max_val:
        tmp = np.random.rayleigh(sigma)
    return tmp
def tab_realisation(n_shear, n_rot, n_rea, n_cas,stamp_xsize, stamp_ysize, psf_stamp_x, psf_stamp_y, mag, n_sersic, r_half, case, path):
    print('case ' + str(case) + ' building table')
    table = Table(names = ['mag', 'n', 'r_half', 'e1', 'e2', 'gamma1', 'bound_x_left', 'bound_x_right', 'bound_y_bottom', 'bound_y_top','pixel_shift_x','pixel_shift_y', 'rotation', 'pixel_noise'], dtype = ['f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'i4', 'i4', 'i4', 'i4', 'f4', 'f4', 'f4', 'i4'], meta = {'n_rot': n_rot,'n_shear': n_shear,'n_rea': n_rea,'n_canc': (n_shear*n_rot*2),'n_cas': n_cas, 'stamp_x': stamp_xsize, 'stamp_y': stamp_ysize, 'psf_x': psf_stamp_x, 'psf_y': psf_stamp_y})
    random_seed = 15783 + case
    rng = np.random.default_rng()
    rotation = np.linspace(0, 180, n_rot, endpoint= False)
    shear = np.linspace(-0.01,0.01 ,n_shear)
    values = []
    count1 = 0
    ud = galsim.UniformDeviate()
    for i in range(n_shear):
        for j in range(n_rot):
            for k in range(2):
                params = [shear[i],rotation[j],k*n_rot*stamp_ysize+j*stamp_xsize + 1, k*n_rot*stamp_ysize+(j+1)*stamp_xsize-1,  i*stamp_ysize + 1, (i+1)*stamp_ysize - 1, k] 
                values.append(params)
            count1 = count1 + 1
    count = 0
    n_canc = n_shear*n_rot*2
    for realisation in range(n_rea):
        dx = (2*ud()-1) * 0.1/2
        dy = (2*ud()-1) * 0.1/2
        e_betrag = trunc_rayleigh(0.25, 0.7, case)
        phi = rng.choice(180)
        for Cancellation in values:
            e1 = e_betrag*np.cos(2*phi) 
            e2 = e_betrag*np.sin(2*phi)
            params = [mag, n_sersic, r_half, e1, e2, Cancellation[0], Cancellation[2], Cancellation[3], realisation*stamp_ysize*n_shear + Cancellation[4], realisation*stamp_ysize*n_shear + Cancellation[5], dx, dy,Cancellation[1], Cancellation[6]]
            table.add_row(params)
        count = count + 1
    if not os.path.isdir(path):
        os.mkdir(path)
    file_name = os.path.join(path, 'Input_data_' + str(case) + '.fits')
    table.write( file_name , overwrite=True) 
    print('case ' + str(case) + ' finished table')
    return None


def training_set_tab(n_shear, n_rot, n_cas, n_rea, stamp_xsize, stamp_ysize, psf_stamp_x, psf_stamp_y, path):
    cat = galsim.Catalog('gems_20090807.fits')
    tab = Table(names = ['mag', 'n', 'r_half'], dtype = ['f4', 'f4', 'f4'])
    for i in range(cat.nobjects):
        if ((cat.get(i, 'GEMS_FLAG')== 4) and (np.abs(cat.get(i, 'ST_MAG_BEST')-cat.get(i, 'ST_MAG_GALFIT')) < 0.5 ) and (20.5 < cat.get(i, 'ST_MAG_GALFIT') < 24.5) and (0.3 < cat.get(i, 'ST_N_GALFIT') < 6.0) and (3.0 < cat.get(i, 'ST_RE_GALFIT') <40.0)):
            params = [cat.get(i, 'ST_MAG_GALFIT'), cat.get(i, 'ST_N_GALFIT'), cat.get(i, 'ST_RE_GALFIT')*0.03]
            tab.add_row(params)
    rng = np.random.default_rng()
    tab_random = rng.choice(tab, size = n_cas)
    final = []
    count = 0
    for case in tab_random:
        mag = case['mag']
        n_sersic = case['n']
        tru_sersicns = np.linspace(0.3, 6.0, 21)
        n_sersic= tru_sersicns[(np.abs(tru_sersicns-n_sersic)).argmin()]
        r_half = case['r_half']
        params = [n_shear, n_rot, n_rea, n_cas,stamp_xsize, stamp_ysize, psf_stamp_x, psf_stamp_y, mag, n_sersic, r_half, count, path]
        final.append(params)
        count = count + 1
    with Pool(processes = 100) as pool:
            pool.starmap(tab_realisation, final)
    return None

def tab_realisation_trys(n_shear, n_rot, n_rea, n_cas,stamp_xsize, stamp_ysize, psf_stamp_x, psf_stamp_y, mag, n_sersic, r_half, case, path):
    print('case ' + str(case) + ' building trys table')
    table = Table(names = ['mag', 'n', 'r_half', 'e1', 'e2', 'gamma1', 'bound_x_left', 'bound_x_right', 'bound_y_bottom', 'bound_y_top','pixel_shift_x','pixel_shift_y', 'rotation', 'pixel_noise'], dtype = ['f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'i4', 'i4', 'i4', 'i4', 'f4', 'f4', 'f4', 'i4'], meta = {'n_rot': n_rot,'n_shear': n_shear,'n_rea': n_rea,'n_canc': (n_shear*n_rot*2),'n_cas': n_cas, 'stamp_x': stamp_xsize, 'stamp_y': stamp_ysize, 'psf_x': psf_stamp_x, 'psf_y': psf_stamp_y})
    random_seed = 15783 + case
    rng = np.random.default_rng()
    rotation = np.linspace(0, 180, n_rot, endpoint= False)
    shear = np.linspace(-0.01,0.01 ,n_shear)
    values = []
    count1 = 0
    ud = galsim.UniformDeviate()
    for i in range(n_shear):
        for j in range(n_rot):
            for k in range(2):
                params = [shear[i],rotation[j],k*n_rot*stamp_ysize+j*stamp_xsize + 1, k*n_rot*stamp_ysize+(j+1)*stamp_xsize-1,  i*stamp_ysize + 1, (i+1)*stamp_ysize - 1, k] 
                values.append(params)
            count1 = count1 + 1
    count = 0
    n_canc = n_shear*n_rot*2
    for realisation in range(n_rea):
        dx = (2*ud()-1) * 0.1/2
        dy = (2*ud()-1) * 0.1/2
        e_betrag = trunc_rayleigh(0.25, 0.7, case)
        phi = rng.choice(180)
        for Cancellation in values:
            e1 = e_betrag*np.cos(2*phi) 
            e2 = e_betrag*np.sin(2*phi)
            params = [mag, n_sersic, r_half, e1, e2, Cancellation[0], Cancellation[2], Cancellation[3], realisation*stamp_ysize*n_shear + Cancellation[4], realisation*stamp_ysize*n_shear + Cancellation[5], dx, dy,Cancellation[1], Cancellation[6]]
            table.add_row(params)
        count = count + 1
    if not os.path.isdir(path):
        os.mkdir(path)
    file_name = os.path.join(path, 'Input_data_' + str(case) + '_' + str(n_rea) +'.fits')
    table.write( file_name , overwrite=True) 
    print('case ' + str(case) + ' finished trys table')
    return None

def training_set_tab_trys(n_shear, n_rot, cases, n_rea, stamp_xsize, stamp_ysize, psf_stamp_x, psf_stamp_y, path):
    final = []
    for i,j in enumerate(cases):
        table = Table.read(path + '/Input_data_' + str(j) + '.fits')
        params = [n_shear, n_rot, n_rea, len(cases),stamp_xsize, stamp_ysize, psf_stamp_x, psf_stamp_y, table['mag'][0], table['n'][0], table['r_half'][0], j, path]
        final.append(params)
    with Pool(processes = 100) as pool:
            pool.starmap(tab_realisation_trys, final)
    return None    

if __name__ == "__main__":
	path = config.workpath('Test')
	training_set_tab(3, 10, 4, 400, 64, 64, 350, 350, path)


# training_set_tab(2, 1, 1, 3, 64, 64,350, 350,  path)
# table = Table.read(path + '/Input_data_0.fits')
# pn = table['pixel_noise']
# print(pn)
# # bound1 = table['bound_x_left']
# ns = table['n']
# psf = table['psf_pol']
# r_half = table['r_half']
# e1 = table['e1']
# e2 = table['e2']
# g1 = table['gamma1']
# r = table['rotation']
# g1 = table['gamma1']
# print(table['gamma1'])
    
def generate_table(ny_tiles, nx_tiles, stamp_xsize, stamp_ysize, path):
    log_format = '%(asctime)s %(filename)s: %(message)s'
    logging.basicConfig(filename='tab.log', format=log_format, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    cat = galsim.Catalog('gems_20090807.fits')
    logging.info('Start')
    logging.info('downloaded GEMS catalog')
    tab = np.zeros((14046,3))
    #print(cat.get(14659,'ST_MAG_GALFIT'))
    count = 0
    N = ny_tiles*nx_tiles
    #rng = galsim.BaseDeviate(random_seed + i + 1)
    #selecting galaxies and storing their parameters 
    for i in range(cat.nobjects):
        if ((cat.get(i, 'GEMS_FLAG')== 4) and (np.abs(cat.get(i, 'ST_MAG_BEST')-cat.get(i, 'ST_MAG_GALFIT')) < 0.5 ) and (20.5 < cat.get(i, 'ST_MAG_GALFIT') < 25.0) and (0.3 < cat.get(i, 'ST_N_GALFIT') < 6.0) and (3.0 < cat.get(i, 'ST_RE_GALFIT') <40.0)):
            tab[count][0] = cat.get(i, 'ST_MAG_GALFIT')  #magnitude
            tab[count][1] = cat.get(i, 'ST_N_GALFIT')  #Sersic index
            tab[count][2] = cat.get(i, 'ST_RE_GALFIT')*0.03  # Half-light radius in arcsec
            count = count + 1
    logging.info('filtered GEMS catalog with cuts')
    rng = np.random.default_rng()
    tab_random = rng.choice(tab, size = N)
    t = Table(names = ['mag', 'n', 'r_half', 'e1', 'e2', 'bound_x_left', 'bound_x_right', 'bound_y_bottom', 'bound_y_top', 'rotation','number'], dtype = ['f4', 'f4', 'f4', 'f4', 'f4', 'i4', 'i4', 'i4', 'i4', 'i4', 'i4'], meta = {'ny_tiles': ny_tiles, 'nx_tiles': ny_tiles, 'n_cas':ny_tiles *nx_tiles,'n_rea':1, 'n_canc':1, 'stamp_x': stamp_xsize, 'stamp_y': stamp_ysize, 'N': N})
    i = 0
    for iy in range(ny_tiles):
        for ix in range(nx_tiles):
            bound1 = ix*stamp_xsize+1
            bound2 = (ix+1)*stamp_xsize-1
            bound3 = iy*stamp_ysize+1
            bound4 = (iy+1)*stamp_ysize-1
            mag = tab_random[i][0]
            n_ser = tab_random[i][1]
            tru_sersicns = np.linspace(0.3, 6.0, 21)
            n_ser= tru_sersicns[(np.abs(tru_sersicns-n_ser)).argmin()]
            r_half = tab_random[i][2]
            e_betrag = trunc_rayleigh(0.25, 0.7, i)  
            phi = rng.choice(180)
            e1 = e_betrag*np.cos(2*phi)
            e2 = e_betrag*np.sin(2*phi)
            params = [mag, n_ser, r_half, e1, e2, bound1, bound2, bound3, bound4, 0, i+1]
            t.add_row(params)
            i = i + 1
    logging.info('%d galaxies and their parameters were drawn randomly from GEMS catalog' %N)
    logging.info('data was written in fits file and can be used to simulate galaxies')
    if not os.path.isdir(path):
            os.mkdir(path)
    file_name = os.path.join(path, 'Input_data.fits')
    t.write( file_name , overwrite=True)  
    return None

def generate_gamma_tab(path, gamma1, gamma2 = 0, psf_pol = 0):
    t = Table(names = ['gamma1', 'gamma2', 'psf_pol'])
    t.add_row([gamma1, gamma2, psf_pol])
    if not os.path.isdir(path):
            os.mkdir(path)
    file_name = os.path.join(path, 'Gamma.fits')
    t.write( file_name , overwrite=True)  

# generate_gamma_tab('Test', 0.2)
#generate_table(1, 1, 60, 60, 'Test')

# table = Table.read('Test/Input_data.fits')

# print(table)
