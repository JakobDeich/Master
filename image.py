import galsim
from astropy.table import Table, Column
import numpy as np
import os
import tab
import time
import logging
import astropy.units
import coord
import config
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

def generate_realisations(path_table, path, case):
    random_seed = 15783
    table = Table.read(path_table)
    stamp_xsize = table.meta['STAMP_X']
    stamp_ysize = table.meta['STAMP_Y']
    n_rea = table.meta['N_REA']
    n_canc = table.meta['N_CANC']
    # print(table['bound_y_bottom'])
    pixel_scale = 0.1 #arcsec/pixel
    pixel_scale_small = 0.02
    mag_sky = 22.35
    Sky_flux = sky_flux(mag_sky)
    gain = 3.1 #e/ADU
    gal_image = galsim.ImageF((stamp_xsize *n_canc*n_rea-1), stamp_ysize-1, scale = pixel_scale)
    #define optical psf with Euclid condition and anisotropy
    psf = generate_psf()
    psf = psf.shear(e1 = table['psf_pol'][0])
    #creating the grid and placing galaxies on it
    count = 0
    count1 = 0
    for Galaxy in table:
        psf_image = galsim.ImageF(stamp_xsize, stamp_ysize, scale = pixel_scale_small)
        psf.drawImage(psf_image)
        file_name = os.path.join(path, 'PSF_' + str(case) + '.fits')
        psf_image.write(file_name)
        break
    for Galaxy in table:
        gal2 = galsim.ImageF(stamp_xsize - 1, stamp_ysize -1, scale = pixel_scale)
        gal_blank = galsim.ImageF(stamp_xsize - 1, stamp_ysize -1, scale = pixel_scale)
        if count%200==0:
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
        #shear galaxy
        gal = gal.shear(g1 = Galaxy['gamma1'])
        #convolve galaxy and psf 
        final_gal = galsim.Convolve([psf,gal], gsparams = gs)
        #add noise
        if Galaxy['pixel_noise'] == 0:
            final_gal.drawImage(sub_gal_image)
            rng = galsim.BaseDeviate(random_seed + count1)
            CCD_Noise = galsim.CCDNoise(rng, sky_level = Sky_flux, gain = gain, read_noise = 4.2)
            sub_gal_image.addNoise(CCD_Noise)
        else:
            final_gal.drawImage(gal2)
            final_gal.drawImage(gal_blank)
            rng = galsim.BaseDeviate(random_seed + count1)
            CCD_Noise = galsim.CCDNoise(rng, sky_level = Sky_flux, gain = gain, read_noise = 4.2)
            gal2.addNoise(CCD_Noise)
            Diff = gal2.array - gal_blank.array
            gal_minus = np.subtract(gal_blank.array,Diff)
            gal_image[b] = galsim.Image(gal_minus)
            #print(sub_gal_image)
            count1 = count1 + 1
        #sub_gal_image.addNoise(CCD_Noise)
        count = count + 1
    if not os.path.isdir(path):
            os.mkdir(path)
    file_name = os.path.join(path, 'Grid_case' + str(case) + '.fits')
    gal_image.write(file_name)
    return None
    

    
# path = config.workpath('Test/')
# # # # tab.training_set_tab(5, 10, 1, 10, 64, 64, path)
# # # # bounds = table['bound_x_left']
# generate_realisations(path + 'Input_data_0.fits', path, 0)

def generate_image(path_table, path):
    log_format = '%(asctime)s %(filename)s: %(message)s'
    logging.basicConfig(filename='image.log', format=log_format, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Start generating grid')
    random_seed = 15783
    table = Table.read(path_table)
    Gamma = Table.read(path + '/Gamma.fits')
    gamma1 = Gamma['gamma1']
    gamma2 = Gamma['gamma2']
    psf_pol = Gamma['psf_pol']
    ny_tiles = table.meta['NY_TILES']
    nx_tiles = table.meta['NX_TILES']
    stamp_xsize = table.meta['STAMP_X']
    stamp_ysize = table.meta['STAMP_Y']
    pixel_scale = 0.1 #arcsec/pixel
    t_exp = 3*565 #s
    gain = 3.1 #e/ADU
    mag_sky = 22.35
    Z_p = 24.6
    l_pix = 0.1 #arcsec
    sky_flux = l_pix**2*t_exp/gain*10**(-0.4*(mag_sky-Z_p))
    gal_image = galsim.ImageF((stamp_xsize * nx_tiles-1)*2+1, stamp_ysize * ny_tiles-1, scale = pixel_scale)
    #define optical psf with Euclid condition and anisotropy
    psf = generate_psf()
    psf = psf.shear(e1 = psf_pol)
    #creating the grid and placing galaxies on it
    count = 0
    for Galaxy in table:
        if count%1000==0:
            print(count)
        flux = gal_flux(Galaxy['mag'])
        #define galaxy with sersic profile
        gs = galsim.GSParams(maximum_fft_size=22000)  #in pixel              
        gal = galsim.Sersic(Galaxy['n'], half_light_radius = Galaxy['r_half'], flux = flux)
        gal = gal.shear(e1=Galaxy['e1'], e2=Galaxy['e2'])
        #create grid
        b = galsim.BoundsI(Galaxy['bound_x_left'], Galaxy['bound_x_right'], Galaxy['bound_y_bottom'], Galaxy['bound_y_top'])
        sub_gal_image = gal_image[b] 
        ud = galsim.UniformDeviate(random_seed + count + 1)
        dx = (2*ud()-1) * pixel_scale/2
        dy = (2*ud()-1) * pixel_scale/2
        #shift galaxies and psfs on the grid
        gal = gal.shift(dx, dy)
        if (Galaxy['rotation'] == 1):
            gal = gal.rotate(galsim.Angle(theta = 90, unit = coord.degrees))
        #shear galaxy
        gal = gal.shear(g1 = gamma1, g2 = gamma2)
        #convolve galaxy and psf 
        final_gal = galsim.Convolve([psf,gal], gsparams = gs)
        # try:
        #     final_gal.drawImage(sub_gal_image)
        # except:
        #     print('Error at galaxy ', count)
        #     logging.error('Error at galaxy %d with elipticities e1 = %f and e2 = %f' %(count, Galaxy['e1'], Galaxy['e2']))
        #add noise
        start1 = time.time()
        final_gal.drawImage(sub_gal_image)
        rng = galsim.BaseDeviate(random_seed + count)
        CCD_Noise = galsim.CCDNoise(rng, sky_level = sky_flux, gain = gain, read_noise = 4.2)
        sub_gal_image.addNoise(CCD_Noise)
        end1 = time.time()
        print(end1-start1)
        count = count + 1
    logging.info('finished generating grid')
    if not os.path.isdir(path):
            os.mkdir(path)
    file_name = os.path.join(path, 'Grid.fits')
    gal_image.write(file_name)
    logging.info('Grid saved in File at %s/Grid.fits' %path)
    return None


# path = config.workpath('Test/')
# table = Table.read(path + '/Input_data_0.fits')
# print(table[500])
# table2 = Table.read('Test/Gamma.fits')
# print(table2)

end = time.time()

total_time = (end - start)/(60*60)  #run time in hours
# print('The system took ', total_time ,' hours to execute the function')