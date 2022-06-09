from astropy.table import Table
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


def generate_table(ny_tiles, nx_tiles, stamp_xsize, stamp_ysize):
    cat = galsim.Catalog('gems_20090807.fits')
    tab = np.zeros((12263,3))
    #print(cat.get(14659,'ST_MAG_GALFIT'))
    count = 0
    N = ny_tiles*nx_tiles
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
    
    bounds1 = np.zeros(N)
    bounds2 = np.zeros(N)
    bounds3 = np.zeros(N)
    bounds4 = np.zeros(N)
    tab_1 = np.zeros(N)
    tab_2 = np.zeros(N)
    tab_3 = np.zeros(N)
    tab_4 = np.zeros(N)
    tab_5 = np.zeros(N)
    for i in range(N):
        tab_1[i] = tab_random[i][0]
        tab_2[i] = tab_random[i][1]
        tru_sersicns = np.linspace(0.3, 6.0, 21)
        tab_2[i] = tru_sersicns[(np.abs(tru_sersicns-tab_2[i])).argmin()]
        tab_3[i] = tab_random[i][2]
        e_betrag = trunc_rayleigh(0.25, 0.7)  
        phi = rng.choice(180)
        e1 = e_betrag*np.cos(2*phi)
        e2 = e_betrag*np.sin(2*phi)
        tab_4[i] = e1
        tab_5[i] = e2   
    
    
    count = 0
    for iy in range(ny_tiles):
        for ix in range(nx_tiles):
            bounds1[count] = ix*stamp_xsize+1
            bounds2[count] = (ix+1)*stamp_xsize-1
            bounds3[count] = iy*stamp_ysize+1
            bounds4[count] = (iy+1)*stamp_ysize-1
            count = count + 1
            
    
    t = Table()
    t = Table([tab_1, tab_2, tab_3, tab_4, tab_5, bounds1, bounds2, bounds3, bounds4], names = ('mag', 'n', 'r_half', 'e1', 'e2', 'bound_x_left', 'bound_x_right', 'bound_y_top', 'bound_y_bottom'), meta = {'ny_tiles': ny_tiles, 'nx_tiles': ny_tiles, 'stamp_xsize': stamp_xsize, 'stamp_ysize': stamp_ysize})
    return t



