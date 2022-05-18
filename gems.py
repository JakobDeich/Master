import galsim
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

random_seed = 15783

cat = galsim.Catalog('gems_20090807.fits')
tab = np.zeros((3,14660))
#print(cat.get(124,'ST_MAG_GALFIT'))
count = 0
#rng = galsim.BaseDeviate(random_seed + i + 1)
#selecting galaxies and storing their parameters 
for i in range(cat.nobjects):
    if ((cat.get(i, 'GEMS_FLAG')== 4) and (np.sqrt((cat.get(i, 'ST_MAG_BEST')-cat.get(i, 'ST_MAG_GALFIT'))**2) < 0.5 ) and (20.5 < cat.get(i, 'ST_MAG_GALFIT') < 25.0) and (0.3 < cat.get(i, 'ST_N_GALFIT') < 6.0) and (0.0 < cat.get(i, 'ST_RE_GALFIT') <40.0)):
        tab[0][count] = cat.get(i, 'ST_MAG_GALFIT')  #magnitude
        tab[1][count] = cat.get(i, 'ST_N_GALFIT')  #Sersic index
        tab[2][count] = cat.get(i, 'ST_RE_GALFIT')  # Half-light radius
        count = count + 1


def trunc_rayleigh(sigma, max_val):
    assert max_val > sigma
    tmp = max_val + 1
    while tmp > max_val:
        tmp = np.random.rayleigh(sigma)
    return tmp
#function for creating a psf with 5 different wavelength in the optical 
def psf(lam):
    optical_psf = galsim.OpticalPSF(lam = lam, diam = 1.2, obscuration = 0.29, nstruts = 6, scale_unit = galsim.arcsec)
    return optical_psf
    


# def Galaxy(lam, mag, n_sersic, r_half):
#     #sigma_w = 0.4
#     t_exp = 3*565 #s
#     gain = 3.1 #e/ADU
#     mag_sky = 22.35
#     Z_p = 24.6
#     l_pix = 0.1 #arcsec
#     e1 = trunc_rayleigh(0.25, 0.7)
#     e2 = trunc_rayleigh(0.25, 0.7)
#     gal_flux = t_exp/gain *10**(-0.4*(mag-Z_p)) 
#     sky_flux = l_pix**2*t_exp/gain*10**(-0.4*(mag_sky-Z_p))
#     #define galaxy with sersic profile
#     gal = galsim.Sersic(n = n_sersic, half_light_radius = r_half, flux = gal_flux)
#     gal = gal.shear(e1=e1, e2=e2)
#     #define optical psf with Euclid condition 
#     image_psf = psf(600)+psf(700)+psf(800)+psf(900)
#     return gal and image_psf


ny_tiles = 122 #122
nx_tiles = 122
stamp_xsize = 40
stamp_ysize = 40
pixel_scale = 0.2 #arcsec/pixel
shift_radius_sq = 0.01
t_exp = 3*565 #s
gain = 3.1 #e/ADU
mag_sky = 22.35
Z_p = 24.6
l_pix = 0.1 #arcsec
def gal_flux(mag): 
    return t_exp/gain *10**(-0.4*(mag-Z_p)) 
sky_flux = l_pix**2*t_exp/gain*10**(-0.4*(mag_sky-Z_p))
gal_image = galsim.ImageF(stamp_xsize * nx_tiles-1, stamp_ysize * ny_tiles-1, scale = pixel_scale)
psf_image = galsim.ImageF(stamp_xsize * nx_tiles-1, stamp_ysize * ny_tiles-1, scale = pixel_scale)

count = 0
#creating the grid and placing galaxies on it
for iy in range(ny_tiles):
    for ix in range(nx_tiles):
        try:
            mag = tab[0][count]
            n_sersic = tab[1][count]
            tru_sersicns = np.linspace(0.3, 6.0, 21)
            tru_sersicn = tru_sersicns[(np.abs(tru_sersicns-n_sersic)).argmin()]
            r_half = tab[2][count]
            if count < 14659:
                count = count + 1
            e1 = trunc_rayleigh(0.25, 0.7)
            e2 = trunc_rayleigh(0.25, 0.7)
            flux = gal_flux(mag)
            #define galaxy with sersic profile
            gal = galsim.Sersic(n = n_sersic, half_light_radius = r_half, flux = flux)
            gal = gal.shear(e1=e1, e2=e2)
            #define optical psf with Euclid condition 
            image_psf = psf(600)+psf(700)+psf(800)+psf(900)
            #create grid
            b = galsim.BoundsI(ix*stamp_xsize+1, (ix+1)*stamp_xsize-1, iy*stamp_ysize+1, (iy+1)*stamp_ysize-1)
            sub_gal_image = gal_image[b] 
            sub_psf_image = psf_image[b]
            rsq = 2 * shift_radius_sq
            ud = galsim.UniformDeviate(random_seed + count + 1)
            while (rsq > shift_radius_sq):
                dx = (2*ud()-1) * shift_radius_sq
                dy = (2*ud()-1) * shift_radius_sq
                rsq = dx**2 + dy**2
            #shift galaxies and psfs on the grid
            gal = gal.shift(dx, dy)
            psf1 = image_psf.shift(dx, dy)
            #convolve galaxy and psf
            final_gal = galsim.Convolve([psf1,gal])
            final_gal.drawImage(sub_gal_image)
            #add noise
            rng = galsim.BaseDeviate(random_seed+count)
            CCD_Noise = galsim.CCDNoise(rng, sky_level = sky_flux, gain = gain, read_noise = 4.2)
            sub_gal_image.addNoise(CCD_Noise)
            psf1.drawImage(sub_psf_image)
        except:
            print(count)
if not os.path.isdir('output'):
    os.mkdir('output')
file_name = os.path.join('output', 'Grid.fits')
file_name_epsf = os.path.join('output','Grid_epsf.fits')
gal_image.write(file_name)
psf_image.write(file_name_epsf)



plt.figure()
# plt.axis([xmin, xmax, ymin, ymax])
plt.imshow(gal_image.array, cmap='gray')
#plt.imshow(image_data_psf.array, cmap='gray', norm=LogNorm())
plt.colorbar(label = 'flux')
plt.xlabel('x-axis [px]')
plt.ylabel('y-axis [px]')
        
    
    
    
    
    