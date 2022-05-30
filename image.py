import galsim
import numpy as np
import os
import tab
import time

start = time.time()

table = tab.generate_table()


f = open('log.txt', 'w')

def psf(lam):
    optical_psf = galsim.OpticalPSF(lam = lam, diam = 1.2, obscuration = 0.29, nstruts = 6, scale_unit = galsim.arcsec)
    return optical_psf

def gal_flux(mag): 
    t_exp = 3*565 #s
    gain = 3.1 #e/ADU
    Z_p = 24.6
    return t_exp/gain *10**(-0.4*(mag-Z_p)) 

def generate_image():
    random_seed = 15783
    ny_tiles = 100 #122
    nx_tiles = 100
    stamp_xsize = 40
    stamp_ysize = 40
    pixel_scale = 0.1 #arcsec/pixel
    pixel_scale_small = 0.02 #arcsec/pixel
    t_exp = 3*565 #s
    gain = 3.1 #e/ADU
    mag_sky = 22.35
    Z_p = 24.6
    l_pix = 0.1 #arcsec
    dummy = 0
    f.write('Test')
    sky_flux = l_pix**2*t_exp/gain*10**(-0.4*(mag_sky-Z_p))
    gal_image = galsim.ImageF(stamp_xsize * nx_tiles-1, stamp_ysize * ny_tiles-1, scale = pixel_scale)
    psf_image = galsim.ImageF(stamp_xsize * nx_tiles-1, stamp_ysize * ny_tiles-1, scale = pixel_scale_small)
    #define optical psf with Euclid condition 
    image_psf = psf(600)+psf(700)+psf(800)+psf(900)
    count = 0
    #creating the grid and placing galaxies on it
    for iy in range(ny_tiles):
        for ix in range(nx_tiles):
            if (count % 100 == 0):
                print(count)
                dummy = repr(count)
                f.write(dummy)
                f.write('\n')
            mag = table[count][0][0]
            n_sersic = table[count][0][1]
            tru_sersicns = np.linspace(0.3, 6.0, 21)
            tru_sersicn = tru_sersicns[(np.abs(tru_sersicns-n_sersic)).argmin()]
            r_half = table[count][0][2]
            e1 = table[count][0][3]
            e2 = table[count][0][4]
            flux = gal_flux(mag)
            #define galaxy with sersic profile
            gs = galsim.GSParams(maximum_fft_size=22000)  #in pixel              
            gal = galsim.Sersic(n = tru_sersicn, half_light_radius = r_half, flux = flux)
            gal = gal.shear(e1=e1, e2=e2)
            #create grid
            b = table[count][1]
            sub_gal_image = gal_image[b] 
            sub_psf_image = psf_image[b]
            ud = galsim.UniformDeviate(random_seed + count + 1)
            dx = (2*ud()-1) * pixel_scale/2
            dy = (2*ud()-1) * pixel_scale/2
            #shift galaxies and psfs on the grid
            gal = gal.shift(dx, dy)
            psf1 = image_psf.shift(dx, dy)
            #convolve galaxy and psf                          
            final_gal = galsim.Convolve([psf1,gal], gsparams = gs)
            try:
                final_gal.drawImage(sub_gal_image)
            except:
                print('Error at galaxy ', count)
                dummy = repr(count)
                f.write(dummy)
            #add noise
            rng = galsim.BaseDeviate(random_seed+count)
            CCD_Noise = galsim.CCDNoise(rng, sky_level = sky_flux, gain = gain, read_noise = 4.2)
            sub_gal_image.addNoise(CCD_Noise)
            psf1.drawImage(sub_psf_image)
            count = count + 1
    if not os.path.isdir('output'):
            os.mkdir('output')
    file_name = os.path.join('output', 'Grid.fits')
    file_name_epsf = os.path.join('output','Grid_epsf.fits')
    gal_image.write(file_name)
    psf_image.write(file_name_epsf)
    return 0

generate_image()

end = time.time()

total_time = (end - start)/(60*60)  #run time in hours
print('The system took ', total_time ,' hours to execute the function')
total_time = repr(total_time)
f.write(total_time)
f.close()