import galsim
from astropy.table import Table, Column
import numpy as np
import os
import random
import tab
import time
import logging
import astropy.units
import coord
import config
import glob
import threading
# start = time.time()

# table = tab.generate_table(5,5,40,40)

def generate_psf():
    psf = galsim.OpticalPSF(lam = 600, diam = 0.6, obscuration = 0.29, nstruts = 6, scale_unit = galsim.arcsec)
    lams = [700,800,900]
    for lam in lams:
        optical_psf = galsim.OpticalPSF(lam = lam, diam = 0.6, obscuration = 0.29, nstruts = 6, scale_unit = galsim.arcsec)
        psf = galsim.Add([psf, optical_psf])
    psf = psf/(len(lams) + 1)
    return psf

def gal_flux(mag): 
    t_exp = 3*565 #s
    gain = 3.1 #e/ADU
    Z_p = 24.6
    return t_exp/gain *10**(-0.4*(mag-Z_p)) 

def sky_flux(mag_sky):
    t_exp = 3*565 #s
    gain = 3.1 #e/ADU
    Z_p = 24.6
    l_pix = 0.1 #arcsec
    sky_flux = l_pix**2*t_exp/gain*10**(-0.4*(mag_sky-Z_p))
    return sky_flux

def generate_psf_image(path):
    table = Table.read(path + '/Input_data.fits')
    Gamma = Table.read(path + '/Gamma.fits')
    stamp_xsize = table.meta['STAMP_X']
    stamp_ysize = table.meta['STAMP_Y']
    e1 = Gamma['psf_pol']
    pixel_scale_small = 0.02 
    psf_image = galsim.ImageF(stamp_xsize, stamp_ysize, scale = pixel_scale_small)
    psf = generate_psf()
    psf = psf.shear(e1 = e1)
    psf.drawImage(psf_image)
    file_name = os.path.join(path, 'PSF.fits')
    psf_image.write(file_name)
    return None

def generate_realisations(path, case):
    random_seed = 15783
    with galsim.utilities.single_threaded():
        rng = galsim.BaseDeviate()
    table = Table.read(path + '/Input_data_' + str(case) + '.fits')
    stamp_xsize = table.meta['STAMP_X']
    stamp_ysize = table.meta['STAMP_Y']
    psf_stamp_x = table.meta['PSF_X']
    psf_stamp_y = table.meta['PSF_Y']
    n_shear = table.meta['N_SHEAR']
    n_rot = table.meta['N_ROT']
    n_rea = table.meta['N_REA']
    # print(table['bound_y_bottom'])
    pixel_scale = 0.1 #arcsec/pixel 
    pixel_scale_small = 0.02
    mag_sky = 22.35
    Sky_flux = sky_flux(mag_sky)
    gain = 3.1 #e/ADU
    gal_image = galsim.ImageF((stamp_xsize *n_rot *2), stamp_ysize *n_shear *n_rea, scale = pixel_scale)
    #creating the grid and placing galaxies on it
    count = 0
    #draw random Euclid PSF and anisotropy
    PSFs_link = sorted(glob.glob('/vol/euclid5/euclid5_raid3/mtewes/Euclid_PSFs_Lance_Jan_2020/f*/sed_true*.fits'))
    PSF_indices = np.arange(len(PSFs_link), dtype=int)
    rng1 = np.random.default_rng()
    PSF_index = rng1.choice(PSF_indices, size = 1)
    PSF = galsim.fits.read(PSFs_link[PSF_index[0]], hdu = 1)
    psf = galsim.InterpolatedImage(PSF, scale = pixel_scale_small)
    psf_image = galsim.ImageF(psf_stamp_x, psf_stamp_y, scale = pixel_scale_small)
    psf.drawImage(psf_image, method = 'no_pixel')
    file_name = os.path.join(path, 'PSF_' + str(case) + '.fits')
    psf_image.write(file_name)
    for Galaxy in table:
        if count%4000==0:
            print(count, case)
        gal_blank = galsim.ImageF(stamp_xsize - 1, stamp_ysize - 1, scale = pixel_scale)
        flux = gal_flux(Galaxy['mag'])
        #define galaxy with sersic profile
        gs = galsim.GSParams(maximum_fft_size=22000)  #in pixel              
        gal = galsim.Sersic(Galaxy['n'], half_light_radius = Galaxy['r_half'], flux = flux)
        gal = gal.shear(e1=Galaxy['e1'], e2=Galaxy['e2'])
        #create grid
        b = galsim.BoundsI(Galaxy['bound_x_left'], Galaxy['bound_x_right'], Galaxy['bound_y_bottom'], Galaxy['bound_y_top'])
        sub_gal_image = gal_image[b] 
        #shift galaxies and psfs on the grid
        gal = gal.shift(Galaxy['pixel_shift_x'], Galaxy['pixel_shift_y'])
        gal = gal.rotate(galsim.Angle(theta = Galaxy['rotation'], unit = coord.degrees))
        #shear galaxy
        gal = gal.shear(g1 = Galaxy['gamma1'])
        #convolve galaxy and psf 
        final_gal = galsim.Convolve([psf,gal], gsparams = gs)
        #add noise
        if Galaxy['pixel_noise'] == 0:
            with galsim.utilities.single_threaded():
                gal_test = galsim.ImageF(stamp_xsize - 1, stamp_ysize - 1, scale = pixel_scale)
                final_gal.drawImage(sub_gal_image)
                final_gal.drawImage(gal_blank)
                final_gal.drawImage(gal_test)
                CCD_Noise = galsim.CCDNoise(rng, sky_level = Sky_flux, gain = gain, read_noise = 4.2)
                sub_gal_image.addNoise(CCD_Noise)
                gal_test.addNoise(CCD_Noise)
                Diff = gal_test.array - gal_blank.array
        else:
            final_gal.drawImage(gal_blank)
            gal_minus = np.subtract(gal_blank.array, Diff)
            gal_image[b] = galsim.Image(gal_minus)
            # nul = (np.subtract(np.add(gal_minus,gal_test.array), 2*gal_blank.array))
            # print(nul)
            # break
        count = count + 1
    print('Galaxies drawn')
    if not os.path.isdir(path):
            os.mkdir(path)
    file_name = os.path.join(path, 'Grid_case' + str(case) + '.fits')
    gal_image.write(file_name)
    return None


def generate_realisations_trys(path, case, n_rea, trys):
    random_seed = 15783
    table = Table.read(path + '/Input_data_' + str(case) + '_' + str(n_rea) +'.fits')
    stamp_xsize = table.meta['STAMP_X']
    stamp_ysize = table.meta['STAMP_Y']
    psf_stamp_x = table.meta['PSF_X']
    psf_stamp_y = table.meta['PSF_Y']
    n_shear = table.meta['N_SHEAR']
    n_rot = table.meta['N_ROT']
    n_rea = table.meta['N_REA']
    # print(table['bound_y_bottom'])
    pixel_scale = 0.1 #arcsec/pixel 
    pixel_scale_small = 0.02
    mag_sky = 22.35
    Sky_flux = sky_flux(mag_sky)
    gain = 3.1 #e/ADU
    gal_image = galsim.ImageF((stamp_xsize *n_rot *2), stamp_ysize *n_shear *n_rea, scale = pixel_scale)
    #creating the grid and placing galaxies on it
    count = 0
    #draw random Euclid PSF and anisotropy
    # PSFs_link = sorted(glob.glob('/vol/euclid5/euclid5_raid3/mtewes/Euclid_PSFs_Lance_Jan_2020/f*/sed_true*.fits'))
    # PSF_indices = np.arange(len(PSFs_link), dtype=int)
    # rng1 = np.random.default_rng()
    # PSF_index = rng1.choice(PSF_indices, size = 1)
    # PSF = galsim.fits.read(PSFs_link[PSF_index[0]], hdu = 1)
    file_name = os.path.join(path, 'PSF_' + str(case) + '.fits')
    PSF = galsim.fits.read(file_name)
    psf = galsim.InterpolatedImage(PSF, scale = pixel_scale_small)
    # psf.drawImage(psf_image, method = 'no_pixel')
    with galsim.utilities.single_threaded():
        rng = galsim.BaseDeviate()
    # psf_image.write(file_name)
    for Galaxy in table:
        gal_blank = galsim.ImageF(stamp_xsize -1, stamp_ysize -1, scale = pixel_scale)
        if count%4000==0:
            print(count, case)
        flux = gal_flux(Galaxy['mag'])
        #define galaxy with sersic profile
        gs = galsim.GSParams(maximum_fft_size=22000)  #in pixel              
        gal = galsim.Sersic(Galaxy['n'], half_light_radius = Galaxy['r_half'], flux = flux)
        gal = gal.shear(e1=Galaxy['e1'], e2=Galaxy['e2'])
        #create grid
        b = galsim.BoundsI(Galaxy['bound_x_left'], Galaxy['bound_x_right'], Galaxy['bound_y_bottom'], Galaxy['bound_y_top'])
        sub_gal_image = gal_image[b] 
        #shift galaxies and psfs on the grid
        gal = gal.shift(Galaxy['pixel_shift_x'], Galaxy['pixel_shift_y'])
        gal = gal.rotate(galsim.Angle(theta = Galaxy['rotation'], unit = coord.degrees))
        #shear galaxy
        gal = gal.shear(g1 = Galaxy['gamma1'])
        #convolve galaxy and psf 
        final_gal = galsim.Convolve([psf,gal], gsparams = gs)
        #add noise
        if Galaxy['pixel_noise'] == 0:
            with galsim.utilities.single_threaded():
                gal_test = galsim.ImageF(stamp_xsize -1, stamp_ysize -1, scale = pixel_scale)
                final_gal.drawImage(sub_gal_image)
                final_gal.drawImage(gal_blank)
                final_gal.drawImage(gal_test)
                CCD_Noise = galsim.CCDNoise(rng, sky_level = Sky_flux, gain = gain, read_noise = 4.2)
                sub_gal_image.addNoise(CCD_Noise)
                gal_test.addNoise(CCD_Noise)
                Diff = gal_test.array - gal_blank.array
        else:
            final_gal.drawImage(gal_blank)
            gal_minus = np.subtract(gal_blank.array, Diff)
            gal_image[b] = galsim.Image(gal_minus)
        count = count + 1
    if not os.path.isdir(path):
            os.mkdir(path)
    # file_name = os.path.join(path, 'Grid_case' + str(case) + '.fits')
    file_name = os.path.join(path, 'Grid_case' + str(case) + '_' + str(trys) + '_' + str(n_rea) +'.fits')
    gal_image.write(file_name)
    return None
    
    
# path = config.workpath('Test/')
# # # tab.training_set_tab(5, 10, 1, 2, 64, 64,350, 350, path)
# # # # # # bounds = table['bound_x_left']
# generate_realisations(path + 'Input_data_0.fits', path, 0)

def generate_image(path):
    log_format = '%(asctime)s %(filename)s: %(message)s'
    logging.basicConfig(filename='image.log', format=log_format, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Start generating grid')
    random_seed = 15783
    table = Table.read(path + 'Input_data.fits')
    ny_tiles = table.meta['NY_TILES']
    nx_tiles = table.meta['NX_TILES']
    stamp_xsize = table.meta['STAMP_X']
    stamp_ysize = table.meta['STAMP_Y']
    pixel_scale = 0.1 #arcsec/pixel
    pixel_scale_small = 0.02
    t_exp = 3*565 #s
    gain = 3.1 #e/ADU
    mag_sky = 22.35
    Z_p = 24.6
    l_pix = 0.1 #arcsec
    sky_flux = l_pix**2*t_exp/gain*10**(-0.4*(mag_sky-Z_p))
    gal_image = galsim.ImageF((stamp_xsize * nx_tiles-1)+1, stamp_ysize * ny_tiles-1, scale = pixel_scale)
    psf_image = galsim.ImageF((stamp_xsize * nx_tiles-1)+1, stamp_ysize * ny_tiles-1, scale = pixel_scale_small)
    #define optical psf with Euclid condition and anisotropy
    #creating the grid and placing galaxies on it
    PSFs_link = sorted(glob.glob('/vol/euclid5/euclid5_raid3/mtewes/Euclid_PSFs_Lance_Jan_2020/f*/sed_true*.fits'))
    PSF_indices = np.arange(len(PSFs_link), dtype=int)
    count = 0
    for Galaxy in table:
        if count%1000==0:
            print(count)
        flux = gal_flux(Galaxy['mag'])
        rng1 = np.random.default_rng()
        PSF_index = rng1.choice(PSF_indices, size = 1)
        PSF = galsim.fits.read(PSFs_link[PSF_index[0]], hdu = 1)
        psf = galsim.InterpolatedImage(PSF, scale = pixel_scale_small)
        #define galaxy with sersic profile
        gs = galsim.GSParams(maximum_fft_size=22000)  #in pixel              
        gal = galsim.Sersic(Galaxy['n'], half_light_radius = Galaxy['r_half'], flux = flux)
        gal = gal.shear(e1=Galaxy['e1'], e2=Galaxy['e2'])
        #create grid
        b = galsim.BoundsI(Galaxy['bound_x_left'], Galaxy['bound_x_right'], Galaxy['bound_y_bottom'], Galaxy['bound_y_top'])
        sub_gal_image = gal_image[b] 
        sub_psf_image = psf_image[b]
        psf.drawImage(sub_psf_image, method = 'no_pixel')
        ud = galsim.UniformDeviate(random_seed + count + 1)
        dx = (2*ud()-1) * pixel_scale/2
        dy = (2*ud()-1) * pixel_scale/2
        #shift galaxies and psfs on the grid
        gal = gal.shift(dx, dy)
        #shear galaxy
        # gal = gal.shear(g1 = gamma1, g2 = gamma2)
        #convolve galaxy and psf 
        final_gal = galsim.Convolve([psf,gal], gsparams = gs)
        final_gal.drawImage(sub_gal_image)
        rng = galsim.BaseDeviate(random_seed + count)
        CCD_Noise = galsim.CCDNoise(rng, sky_level = sky_flux, gain = gain, read_noise = 4.2)
        sub_gal_image.addNoise(CCD_Noise)
        count = count + 1
    logging.info('finished generating grid')
    if not os.path.isdir(path):
            os.mkdir(path)
    file_name = os.path.join(path, 'Grid.fits')
    gal_image.write(file_name)
    file_name = os.path.join(path, 'PSFs.fits')
    psf_image.write(file_name)
    logging.info('Grid saved in File at %s/Grid.fits' %path)
    return None

if __name__ == "__main__":
    path = config.workpath('Test/')
    # # tab.training_set_tab(5, 10, 1, 2, 64, 64,350, 350, path)
    # # # # # bounds = table['bound_x_left']
    generate_realisations(path, 1)
    # path = config.workpath('Comparison/')
    # tab.generate_table(5, 5, 64, 64, path)
    # generate_image(path)
    
# path = config.workpath('Test/')
# table = Table.read(path + '/Input_data_0.fits')
# print(table[500])
# table2 = Table.read('Test/Gamma.fits')
# print(table2)

# end = time.time()

# total_time = (end - start)/(60*60)  #run time in hours
# print('The system took ', total_time ,' hours to execute the function')
