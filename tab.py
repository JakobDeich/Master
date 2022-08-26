
from astropy.table import Table
from astropy.io import ascii
import numpy as np
import galsim
import random
import os
import logging
import time
#import numpy.random.Generator

def trunc_rayleigh(sigma, max_val):
    assert max_val > sigma
    tmp = max_val + 1
    while tmp > max_val:
        tmp = np.random.rayleigh(sigma)
    return tmp


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
    t = Table(names = ['mag', 'n', 'r_half', 'e1', 'e2', 'bound_x_left', 'bound_x_right', 'bound_y_bottom', 'bound_y_top', 'rotation','number'], dtype = ['f4', 'f4', 'f4', 'f4', 'f4', 'i4', 'i4', 'i4', 'i4', 'i4', 'i4'], meta = {'ny_tiles': ny_tiles, 'nx_tiles': ny_tiles, 'stamp_x': stamp_xsize, 'stamp_y': stamp_ysize, 'N': N})
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
            e_betrag = trunc_rayleigh(0.25, 0.7)  
            phi = rng.choice(180)
            e1 = e_betrag*np.cos(2*phi)
            e2 = e_betrag*np.sin(2*phi)
            params = [mag, n_ser, r_half, e1, e2, bound1, bound2, bound3, bound4, 0, i+1]
            t.add_row(params)
            params = [mag, n_ser, r_half, e1, e2, stamp_xsize* nx_tiles + bound1,stamp_xsize* nx_tiles + bound2, bound3, bound4, 1, i+1]
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