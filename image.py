import galsim
from astropy.table import Table
import numpy as np
import os
import tab
import time
import logging

start = time.time()

# table = tab.generate_table(5,5,40,40)

def generate_psf():
    psf = galsim.OpticalPSF(lam = 600, diam = 1.2, obscuration = 0.29, nstruts = 6, scale_unit = galsim.arcsec)
    for lam in [700,800,900]:
        optical_psf = galsim.OpticalPSF(lam = lam, diam = 1.2, obscuration = 0.29, nstruts = 6, scale_unit = galsim.arcsec)
        psf = galsim.Add([psf, optical_psf])
    return psf

def gal_flux(mag): 
    t_exp = 3*565 #s
    gain = 3.1 #e/ADU
    Z_p = 24.6
    return t_exp/gain *10**(-0.4*(mag-Z_p)) 

def generate_image(path_table, path):
    log_format = '%(asctime)s %(filename)s: %(message)s'
    logging.basicConfig(filename='image.log', format=log_format, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Start generating grid')
    random_seed = 15783
    table = Table.read(path_table)
    ny_tiles = table.meta['NY_TILES']
    nx_tiles = table.meta['NX_TILES']
    stamp_xsize = table.meta['STAMP_X']
    stamp_ysize = table.meta['STAMP_Y']
    pixel_scale = 0.1 #arcsec/pixel
    pixel_scale_small = 0.02 #arcsec/pixel
    t_exp = 3*565 #s
    gain = 3.1 #e/ADU
    mag_sky = 22.35
    Z_p = 24.6
    l_pix = 0.1 #arcsec
    sky_flux = l_pix**2*t_exp/gain*10**(-0.4*(mag_sky-Z_p))
    gal_image = galsim.ImageF(stamp_xsize * nx_tiles-1, stamp_ysize * ny_tiles-1, scale = pixel_scale)
    #psf_image = galsim.ImageF(stamp_xsize * nx_tiles-1, stamp_ysize * ny_tiles-1, scale = pixel_scale_small)
    #define optical psf with Euclid condition 
    psf = generate_psf()
    #creating the grid and placing galaxies on it
    count = 0
    for Galaxy in table:
        if count%1==0:
            print(count)
        flux = gal_flux(Galaxy['mag'])
        #define galaxy with sersic profile
        gs = galsim.GSParams(maximum_fft_size=22000)  #in pixel              
        gal = galsim.Sersic(Galaxy['n'], half_light_radius = Galaxy['r_half'], flux = flux)
        gal = gal.shear(e1=Galaxy['e1'], e2=Galaxy['e2'])
        #create grid
        b = galsim.BoundsI(Galaxy['bound_x_left'], Galaxy['bound_x_right'], Galaxy['bound_y_top'], Galaxy['bound_y_bottom'])
        sub_gal_image = gal_image[b] 
        #sub_psf_image = psf_image[b]
        ud = galsim.UniformDeviate(random_seed + count + 1)
        dx = (2*ud()-1) * pixel_scale/2
        dy = (2*ud()-1) * pixel_scale/2
        #shift galaxies and psfs on the grid
        gal = gal.shift(dx, dy)
        #psf1 = image_psf.shift(dx, dy)
        #convolve galaxy and psf                          
        final_gal = galsim.Convolve([psf,gal], gsparams = gs)
        try:
            final_gal.drawImage(sub_gal_image)
        except:
            print('Error at galaxy ', count)
            logging.error('Error at galaxy %d with elipticities e1 = %f and e2 = %f' %count %Galaxy['e1'] % Galaxy['e2'])
        #add noise
        rng = galsim.BaseDeviate(random_seed + count)
        CCD_Noise = galsim.CCDNoise(rng, sky_level = sky_flux, gain = gain, read_noise = 4.2)
        sub_gal_image.addNoise(CCD_Noise)
        count = count + 1
        #psf1.drawImage(sub_psf_image)
    logging.info('finished generating grid')
    if not os.path.isdir(path):
            os.mkdir(path)
    file_name = os.path.join(path, 'Grid.fits')
    #file_name_epsf = os.path.join('output','Grid_epsf.fits')
    gal_image.write(file_name)
    #psf_image.write(file_name_epsf)
    logging.info('Grid saved in File at %s /Grid.fits' %path)
    return None

#generate_image('Test/table.fits', 'output')

end = time.time()

total_time = (end - start)/(60*60)  #run time in hours
print('The system took ', total_time ,' hours to execute the function')