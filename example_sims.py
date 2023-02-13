import galsim
import numpy as np
import image
import config
import os
import glob
import ksb
from astropy.table import Table
import matplotlib.pyplot as plt

def PSF(file_name, trys = 0):
    dic = {}
    sigw_sub = 10
    # path = config.workpath('Example')
    PSF = galsim.fits.read(file_name, hdu = 1)
    psf = galsim.InterpolatedImage(PSF, scale = 0.02)
    psf_image = galsim.ImageF(350, 350, scale = 0.02)
    psf.drawImage(psf_image, method = 'no_pixel')
    meas_on_psf = ksb.ksb_moments(psf_image.array, sigw = sigw_sub)
    dic['e1'] = meas_on_psf['e1']
    dic['e2'] = meas_on_psf['e2']
    return dic
    # file_name = os.path.join(path, 'PSF_' + str(trys) + '.fits')
    # psf_image.write(file_name)


def example_sim():
    # stamp_xsize = 350
    # stamp_ysize = 350
    # pixel_scale = 0.02 #arcsec/pixel
    # pixel_scale_small = 0.02
    # t_exp = 3*565 #s
    # gain = 3.1 #e/ADU
    # mag_sky = 22.35
    # Z_p = 24.6
    # l_pix = 0.1 #arcsec
    # sky_flux = l_pix**2*t_exp/gain*10**(-0.4*(mag_sky-Z_p))
    # gal_image = galsim.ImageF(stamp_xsize, stamp_ysize, scale = pixel_scale)
    # gal_image_conv = galsim.ImageF(stamp_xsize, stamp_ysize, scale = pixel_scale)
    nums = [2,4,6,7,9]
    ess = []
    for i in nums:
        PSFs_link = sorted(glob.glob('/vol/euclid5/euclid5_raid3/mtewes/Euclid_PSFs_Lance_Jan_2020/f' + str(i) +'/sed_true*.fits'))
        es1 =[]
        es2 = []
        for i in range(300):
            PSF_indices = np.arange(len(PSFs_link), dtype=int)
            rng1 = np.random.default_rng()
            PSF_index = rng1.choice(PSF_indices, size = 1)
            dic = PSF(PSFs_link[PSF_index[0]])
            es1.append(dic['e1'])
            es2.append(dic['e2'])
        ess.append([np.mean(es1), np.mean(es2)])
    print(ess)
    return None
        
    
cat = galsim.Catalog('gems_20090807.fits')
tab = Table(names = ['mag', 'n', 'r_half'], dtype = ['f4', 'f4', 'f4'])
for i in range(cat.nobjects):
    if ((cat.get(i, 'GEMS_FLAG')== 4) and (np.abs(cat.get(i, 'ST_MAG_BEST')-cat.get(i, 'ST_MAG_GALFIT')) < 0.5 ) and (20.5 < cat.get(i, 'ST_MAG_GALFIT') < 24.5) and (0.3 < cat.get(i, 'ST_N_GALFIT') < 6.0) and (3.0 < cat.get(i, 'ST_RE_GALFIT') <40.0)):
         params = [cat.get(i, 'ST_MAG_GALFIT'), cat.get(i, 'ST_N_GALFIT'), cat.get(i, 'ST_RE_GALFIT')*(12/40)]
         tab.add_row(params)  
print(len(cat), len(tab), len(tab)/len(cat))

# plt.scatter(tab['mag'], tab['r_half'], marker = '.', s = 3, color = 'black')
# plt.xlabel('Magnitude')
# plt.ylabel('Half-light radius [VIS pix]')
plt.hist(tab['mag'], bins = 100, color = 'grey')
    
    # mag = 23
    # r_half = 0.4
    # n = 2
    # flux = image.gal_flux(mag)
    # # gal = galsim.Sersic(n, half_light_radius = r_half, flux = flux)
    # gal = galsim.TopHat(radius = 0.3)
    # # gal = gal.shear(e1=0., e2=0.1)
    # gal.drawImage(gal_image)
    # lam = 700  # nm

    # diam = 2   # meters

    # lam_over_diam = (lam * 1.e-9) / diam  # radians

    # lam_over_diam *= 206265  # Convert to arcsec

    # airy = galsim.Airy(lam=lam, diam=diam, scale_unit=galsim.arcsec)
    # psf = image.generate_psf()
    # psf.drawImage(psf_image)
    # final_gal = galsim.Convolve([psf,gal])
    # final_gal.drawImage(gal_image_conv)
    # gal_image.write(config.workpath('Example') + '/gal.fits')
    # psf_image.write(config.workpath('Example') + '/psf_airy.fits')
    # gal_image_conv.write(config.workpath('Example') + '/gal_conv.fits')

# example_sim()

