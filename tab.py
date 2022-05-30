from astropy.table import QTable
import astropy.units as u
import numpy as np
import galsim
import random
#import numpy.random.Generator

def trunc_rayleigh(sigma, max_val):
    assert max_val > sigma
    tmp = max_val + 1
    while tmp > max_val:
        tmp = np.random.rayleigh(sigma)
    return tmp


def generate_table():
    cat = galsim.Catalog('gems_20090807.fits')
    tab = np.zeros((12263,3))
    #print(cat.get(14659,'ST_MAG_GALFIT'))
    count = 0
    N = 10000
    ny_tiles = 100 #122
    nx_tiles = 100
    stamp_xsize = 40
    stamp_ysize = 40 
    #rng = galsim.BaseDeviate(random_seed + i + 1)
    #selecting galaxies and storing their parameters 
    for i in range(cat.nobjects):
        if ((cat.get(i, 'GEMS_FLAG')== 4) and (np.abs(cat.get(i, 'ST_MAG_BEST')-cat.get(i, 'ST_MAG_GALFIT')) < 0.5 ) and (20.5 < cat.get(i, 'ST_MAG_GALFIT') < 25.0) and (0.3 < cat.get(i, 'ST_N_GALFIT') < 4.0) and (5.0 < cat.get(i, 'ST_RE_GALFIT') <40.0)):
            tab[count][0] = cat.get(i, 'ST_MAG_GALFIT')  #magnitude
            tab[count][1] = cat.get(i, 'ST_N_GALFIT')  #Sersic index
            tab[count][2] = cat.get(i, 'ST_RE_GALFIT')  # Half-light radius
            count = count + 1
    rng = np.random.default_rng()
    tab_random = rng.choice(tab, size = N)
    
    bounds = []
    tab_fin = np.zeros((N,5))
    for i in range(N):
        tab_fin[i][0] = tab_random[i][0]
        tab_fin[i][1] = tab_random[i][1]
        tab_fin[i][2] = tab_random[i][2]
        e_betrag = trunc_rayleigh(0.25, 0.7)  
        phi = 180* random.random()
        e1 = e_betrag*np.cos(2*phi)
        e2 = e_betrag*np.sin(2*phi)
        tab_fin[i][3] = e1
        tab_fin[i][4] = e2   
    
    
    
    for iy in range(ny_tiles):
        for ix in range(nx_tiles):
            bounds.append(galsim.BoundsI(ix*stamp_xsize+1, (ix+1)*stamp_xsize-1, iy*stamp_ysize+1, (iy+1)*stamp_ysize-1))
    
    t = QTable([tab_fin, bounds])
    return t


