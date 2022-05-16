import numpy as np
from astropy.io import fits
import sys
import os
import math
import galsim
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)


##############################################################################
# Use `astropy.io.fits.info()` to display the structure of the file:

# fits.info(image_file)
# fits.info(image_file_psf)

##############################################################################
# Generally the image information is located in the Primary HDU, also known
# as extension 0. Here, we use `astropy.io.fits.getdata()` to read the image
# data from this first extension using the keyword argument ``ext=0``:


# # #print(image_data)
# gal_flux = 1.e5    # counts
# gal_r0 = 0.3       # arcsec
# g1 = 0.1           #
# g2 = 0.2           #
# psf_beta = 5       #
# psf_re = 0.5      # arcsec
# pixel_scale = 0.1  # arcsec / pixel
# sky_level = 2.5e3  # counts / arcsec^2
# e1 = 0.6
# e2 = 0.0
# random_seed = 1534225


# gal = galsim.Exponential(flux=gal_flux, scale_radius=gal_r0)
# gal = gal.shear(e1=e1, e2=e2)
# # Define the PSF profile.
# psf = galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=psf_re)

# # Final profile is the convolution of these.
# final = galsim.Convolve([gal, psf])
                        
# # Draw the image with a particular pixel scale.
# image = gal.drawImage(scale=pixel_scale)
# image = final.drawImage(scale=pixel_scale)
# # The "effective PSF" is the PSF as drawn on an image, which includes the convolution
# # by the pixel response.  We label it epsf here.
# image_epsf = psf.drawImage(scale=pixel_scale)

# # To get Poisson noise on the image, we will use a class called PoissonNoise.
# # However, we want the noise to correspond to what you would get with a significant
# # flux from tke sky.  This is done by telling PoissonNoise to add noise from a
# # sky level in addition to the counts currently in the image.
# rng = galsim.BaseDeviate(random_seed+1)
# # One wrinkle here is that the PoissonNoise class needs the sky level in each pixel,
# # while we have a sky_level in counts per arcsec^2.  So we need to convert:
# sky_level_pixel = sky_level * pixel_scale**2
# #noise = galsim.PoissonNoise(rng, sky_level=sky_level_pixel)
# noise = galsim.GaussianNoise(rng, sigma = 0.01)
# image.addNoise(noise)

# # Write the image to a file.
# if not os.path.isdir('output'):
#     os.mkdir('output')
# file_name = os.path.join('output', 'demo2.fits')
# file_name_epsf = os.path.join('output','demo2_epsf.fits')
# image.write(file_name)
# image_epsf.write(file_name_epsf)

# #results = galsim.hsm.EstimateShear(image, image_epsf)
# #print(results)

# # #exp_shear = galsim.Shear(g1=g1, g2=g2)



# image_file = '/Users/JakobD/Desktop/UniBonn/Semester 4/Masterarbeit/output/demo2.fits'
# image_file_psf = '/Users/JakobD/Desktop/UniBonn/Semester 4/Masterarbeit/output/demo2_epsf.fits'

# image_data_gal = fits.getdata(image_file, ext=0)
# image_data_psf = fits.getdata(image_file_psf, ext=0)

def psf(lam):
    optical_psf = galsim.OpticalPSF(lam = lam, diam = 1.2, obscuration = 0.29, nstruts = 6, scale_unit = galsim.arcsec)
    return optical_psf
    

def Galaxy(lam):
    n = 2
    r_half = 0.3
    #sigma_w = 0.4
    t_exp = 3*565 #s
    gain = 3.1 #e/ADU
    mag = 22.5
    mag_sky = 22.35
    Z_p = 24.6
    l_pix = 0.1 #arcsec
    gal_flux = t_exp/gain *10**(-0.4*(mag-Z_p)) 
    sky_flux = l_pix**2*t_exp/gain*10**(-0.4*(mag_sky-Z_p))
    g1 = 0.4
    g2 = 0.4
    pixel_scale = 0.2 #arcsec/pixel
    pixel_scale_psf = 0.0125 #arcsec/pixel
    random_seed = 1534225
    #define galaxy with sersic profile
    gal = galsim.Sersic(n = n, half_light_radius = r_half, flux = gal_flux)
    gal = gal.shear(g1=g1, g2=g2)
    #define optical psf with Euclid condition 
    image_psf = psf(600)+psf(700)+psf(800)+psf(900)
    #convolve PSF and galaxy
    final = galsim.Convolve([gal , image_psf])
    #image = gal.drawImage(scale = pixel_scale)
    image = final.drawImage(scale = pixel_scale)
    image_psf = image_psf.drawImage(scale = pixel_scale_psf)
    
    #simulate CCD noise
    rng = galsim.BaseDeviate(random_seed+1)
    CCD_Noise = galsim.CCDNoise(rng, sky_level = sky_flux, gain = gain, read_noise = 4.2)
    #CCD_Noise = galsim.GaussianNoise(rng, sigma_w)
    image.addNoise(CCD_Noise)
    
    # Write the image to a file.
    if not os.path.isdir('output'):
        os.mkdir('output')
    file_name = os.path.join('output', 'EuclidGal.fits')
    file_name_epsf = os.path.join('output','EuclidGal_epsf.fits')
    image.write(file_name)
    image_psf.write(file_name_epsf)
    return g1 and g2

Galaxy(lam=600)
    
image_file = 'output/EuclidGal.fits'
image_file_psf = 'output/EuclidGal_epsf.fits'

image_data_gal = galsim.fits.read(image_file)
image_data_psf = galsim.fits.read(image_file_psf)
image_data_gal = image_data_gal.subsample(nx = 4,ny = 4)

def gauss2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):

        return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))



def ksb_moments(stamp,xc=None,yc=None,sigw=2.0,prec=0.01):

        stamp_size=stamp.shape[0]

        # initialize the grid
        y,x=np.mgrid[:stamp_size,:stamp_size]

        dx=1.0
        dy=1.0

        if xc==None:
            xc=stamp_size/2
        if yc==None:
            yc=stamp_size/2


        # first recenter the weight function

        ntry=0
        while (abs(dx)>prec and abs(dy)>prec and ntry<30):

                w = gauss2d(x, y, xc, yc, sigw, sigw)
                ftot=np.sum(w*stamp)

                if ftot>0:

                        wx= x-xc
                        wy= y-yc

                        dx,dy=np.sum(w*wx*stamp)/ftot, np.sum(w*wy*stamp)/ftot
                        xc=xc+dx
                        yc=yc+dy

                ntry=ntry+1

        #print("ntry: ", ntry)

        # compute the polarisation

        w = gauss2d(x, y, xc, yc, sigw, sigw)

        xx= np.power(x-xc,2)
        xy= np.multiply(x-xc,y-yc)
        yy= np.power(y-yc,2)

        q11=np.sum(xx*w*stamp)
        q12=np.sum(xy*w*stamp)
        q22=np.sum(yy*w*stamp)

        denom= q11 + q22

        if denom!=0.0:

                e1=(q11 - q22) / denom
                e2=(2. * q12) / denom

                # compute KSB polarisabilities
                # need to precompute some of this and repeat to speed up

                wp= -0.5 / sigw**2 * w
                wpp= 0.25 / sigw**4 * w

                DD = xx + yy
                DD1 = xx - yy
                DD2 = 2. * xy

                Xsm11= np.sum((2. * w + 4. * wp * DD + 2. * wpp * DD1 * DD1) * stamp)
                Xsm22= np.sum((2. * w + 4. * wp * DD + 2. * wpp * DD2 * DD2) * stamp)
                Xsm12= np.sum((2. * wpp * DD1 * DD2) * stamp)
                Xsh11= np.sum((2. * w * DD + 2. * wp * DD1 * DD1) * stamp)
                Xsh22= np.sum((2. * w * DD + 2. * wp * DD2 * DD2) * stamp)
                Xsh12= np.sum((2. * wp * DD1 * DD2) * stamp)

                em1 = np.sum((4. * wp + 2. * wpp * DD) * DD1 * stamp) / denom
                em2 = np.sum((4. * wp + 2. * wpp * DD) * DD2 * stamp) / denom
                eh1 = np.sum(2. * wp * DD * DD1 * stamp) / denom + 2. * e1
                eh2 = np.sum(2. * wp * DD * DD2 * stamp) / denom + 2. * e2

                psm11= Xsm11/ denom - e1 * em1
                psm22= Xsm11/ denom - e2 * em2
                psm12= Xsm12/ denom - 0.5 * (e1 * em2 + e2 * em1)
                #here multiply with bsm? 
                
                
                psh11= Xsh11 / denom - e1 * eh1
                psh22= Xsh22 / denom - e2 * eh2
                psh12= Xsh12 / denom - 0.5 * (e1 * eh2 + e2 * eh1)

                ksbpar={}
                ksbpar['xc']=xc
                ksbpar['yc']=yc
                ksbpar['e1']=e1
                ksbpar['e2']=e2
                ksbpar['Psm11']=psm11
                ksbpar['Psm22']=psm22
                ksbpar['Psh11']=psh11
                ksbpar['Psh22']=psh22

                return ksbpar


# Und so wendet man es dann z.B. an:
    
# print(image_data_gal.shape)
# print(image_data_psf.shape)

##############################################################################
# Display the image data:
from matplotlib.colors import LogNorm
# xmin = 70
# ymin = 70
# xmax = 110
# ymax = 110
plt.figure()
# plt.axis([xmin, xmax, ymin, ymax])
plt.imshow(image_data_gal.array, cmap='gray')
#plt.imshow(image_data_psf.array, cmap='gray', norm=LogNorm())
plt.colorbar(label = 'flux')
plt.xlabel('x-axis [px]')
plt.ylabel('y-axis [px]')
plt.title('shear $g_1$ = 0.4 and $g_2$ = 0.4')
# meas_on_galaxy = ksb_moments(image_data_gal.array)
# #print(meas_on_galaxy)
# meas_on_psf = ksb_moments(image_data_psf.array)

# e1_anisotropy_correction = (meas_on_galaxy['Psm11'] * (meas_on_psf['e1']/meas_on_psf['Psm11']))
# e2_anisotropy_correction = (meas_on_galaxy['Psm22'] * (meas_on_psf['e2']/meas_on_psf['Psm22']))

# print(e1_anisotropy_correction, e2_anisotropy_correction, meas_on_galaxy)
#print(meas_on_galaxy)
#print((meas_on_galaxy['e2'] - (meas_on_psf['Psm22'] * meas_on_psf['e2']))/meas_on_galaxy['Psm22'])


