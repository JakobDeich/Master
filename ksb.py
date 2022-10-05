import numpy as np
from astropy.io import fits
import sys
import os
import math
import logging 
import galsim
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
#import image
import tab
import config
import image
from astropy.table import Table, Column
from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry
from photutils.morphology import data_properties

# table = tab.generate_table()
# #image.generate_image()
# image_file = 'output/Grid.fits'
# image_file_psf = 'output/Grid_epsf.fits'

# gal_image = galsim.fits.read(image_file)
# psf_image = galsim.fits.read(image_file_psf)
# gal_image = gal_image.subsample(nx = 5,ny = 5)

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
                psm22= Xsm22/ denom - e2 * em2
                psm12= Xsm12/ denom - 0.5 * (e1 * em2 + e2 * em1)
                
                
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
                ksbpar['Psm12']=psm12
                ksbpar['Psh11']=psh11
                ksbpar['Psh22']=psh22
                ksbpar['Psh12']=psh12
                return ksbpar
   
path = config.workpath('Test')
table = Table.read(path + '/Input_data_8.fits')
PSF = galsim.fits.read(path + '/PSF_8.fits')
meas_on_psf = ksb_moments(PSF.array, sigw = 10)
print(table['psf_pol'])
image_file = path + '/Grid_case8.fits'
gal_image = galsim.fits.read(image_file)
a = []
c = []
for Galaxy in table:
    if Galaxy['pixel_noise']==1:
        b = galsim.BoundsI(Galaxy['bound_x_left'], Galaxy['bound_x_right'], Galaxy['bound_y_bottom'], Galaxy['bound_y_top'])
        sub_gal_image = gal_image[b] 
        sub_gal_image = sub_gal_image.subsample(nx = 5,ny = 5)
        meas_on_galaxy = ksb_moments(sub_gal_image.array, sigw = 10)
        e1_anisotropy_correction = (meas_on_galaxy['Psm11'] * (meas_on_psf['e1']/meas_on_psf['Psm11']))
        a.append(meas_on_galaxy['e1'])
    else:
        b = galsim.BoundsI(Galaxy['bound_x_left'], Galaxy['bound_x_right'], Galaxy['bound_y_bottom'], Galaxy['bound_y_top'])
        sub_gal_image = gal_image[b] 
        sub_gal_image = sub_gal_image.subsample(nx = 5,ny = 5)
        meas_on_galaxy = ksb_moments(sub_gal_image.array, sigw = 10)
        e1_anisotropy_correction = (meas_on_galaxy['Psm11'] * (meas_on_psf['e1']/meas_on_psf['Psm11']))
        c.append(meas_on_galaxy['e1'])
print(len(a), len(c))        
    # e1_anisotropy_correction = (meas_on_galaxy['Psm11'] * (meas_on_psf['e1']/meas_on_psf['Psm11']))
    # e2_anisotropy_correction = (meas_on_galaxy['Psm22'] * (meas_on_psf['e2']/meas_on_psf['Psm22']))
def calculate_ksb(path):
    sigw_sub = 10 
    image.generate_psf_image(path)
    PSF = galsim.fits.read(path + '/PSF.fits')
    meas_on_psf = ksb_moments(PSF.array, sigw = sigw_sub)
    log_format = '%(asctime)s %(filename)s: %(message)s'
    logging.basicConfig(filename='ksb.log', format=log_format, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    table = Table.read(path + '/Input_data.fits')
    logging.info('Read in table from %s /table.fits' %path)
    image_file = path + '/Grid.fits'
    gal_image = galsim.fits.read(image_file)
    stamp_x_size = table.meta['STAMP_X']
    stamp_y_size = table.meta['STAMP_Y']
    tab = []
    tab2 = []
    tab3 = []
    tab4 = []
    tab5 = []
    tab6 = []
    tab7 = []
    tab8 = []
    logging.info('start ksb algorithm')
    count = 0
    positions = [((stamp_x_size*5)/2., (stamp_y_size*5)/2.)]
    aperture = CircularAperture(positions, r=sigw_sub)
    aperture2 =  CircularAperture(positions, r=sigw_sub*1.5)
    for Galaxy in table:        
        if count%1000==0:
            logging.warning('Galaxy number %i' %count)
            print(str(count) + ' Galaxies examined')
        b = galsim.BoundsI(Galaxy['bound_x_left'], Galaxy['bound_x_right'], Galaxy['bound_y_bottom'], Galaxy['bound_y_top'])
        sub_gal_image = gal_image[b] 
        sub_gal_image = sub_gal_image.subsample(nx = 5,ny = 5)
        meas_on_galaxy = ksb_moments(sub_gal_image.array, sigw = sigw_sub)
        e1_anisotropy_correction = (meas_on_galaxy['Psm11'] * (meas_on_psf['e1']/meas_on_psf['Psm11']))
        #e2_anisotropy_correction = (meas_on_galaxy['Psm22'] * (meas_on_psf['e2']/meas_on_psf['Psm22']))
        P_g11      = meas_on_galaxy['Psh11']-meas_on_galaxy['Psm11']/meas_on_psf['Psm11']*meas_on_psf['Psh11']
        #P_g22 = meas_on_galaxy['Psh22']-meas_on_galaxy['Psm22']/meas_on_psf['Psm22']*meas_on_psf['Psh22']
        phot_table = aperture_photometry(sub_gal_image.array, aperture)
        phot_table2 = aperture_photometry(sub_gal_image.array, aperture2)
        cat = data_properties(sub_gal_image.array)
        columns = ['label', 'xcentroid', 'ycentroid', 'semimajor_sigma','semiminor_sigma', 'orientation']
        tbl = cat.to_table(columns=columns)
        a = tbl['semimajor_sigma'] 
        b = tbl['semiminor_sigma'] 
        my_moments = galsim.hsm.FindAdaptiveMom(sub_gal_image, guess_sig = 10, strict = (False))
        Radius = np.sqrt( a * b )
        tab.append(Radius)
        tab2.append(P_g11)
        tab3.append(e1_anisotropy_correction)
        tab4.append(meas_on_galaxy['e1'])
        tab5.append(phot_table['aperture_sum'])
        if phot_table['aperture_sum'] > 600:
            tab6.append(0)
        else:
            tab6.append(1)
        tab7.append(my_moments.moments_sigma)
        tab8.append(my_moments.moments_rho4)
        count = count + 1 
    Col_A = Column(name = 'Pg_11', data = tab2)
    Col_B = Column(name = 'anisotropy_corr', data = tab3)
    Col_C = Column(name = 'e1_cal', data = tab4)
    Col_D = Column(name = 'aperture_sum', data = tab5)
    Col_E = Column(name = 'radius_est', data = tab)
    Col_F = Column(name = 'Flag', data = tab6)
    Col_G = Column(name = 'sigma_mom', data = tab7)
    Col_H = Column(name = 'rho4_mom', data = tab8)
    try:
        table.add_columns([Col_A, Col_B, Col_C, Col_D, Col_E, Col_F, Col_G, Col_H])
        logging.info('Add columns to table')
    except:
        table.replace_column(name = 'Pg_11', col = Col_A)
        table.replace_column(name = 'anisotropy_corr', col = Col_B)
        table.replace_column(name = 'e1_cal', col = Col_C)
        table.replace_column(name = 'aperture_sum', col = Col_D)
        table.replace_column(name = 'radius_est', col = Col_E)
        table.replace_column(name = 'Flag', col = Col_F)
        table.replace_column(name = 'sigma_mom', col = Col_G)
        table.replace_column(name = 'rho4_mom', col = Col_H)
        logging.info('replaced columns in table')
    table.write( path + '/Measured_ksb.fits' , overwrite=True) 
    logging.info('overwritten old table with new table including e1_cal and Pg_11')
    return None

def calculate_ksb_training(path, case):
    sigw_sub = 10 
    path = config.workpath(path)
    log_format = '%(asctime)s %(filename)s: %(message)s'
    logging.basicConfig(filename='ksb.log', format=log_format, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    table = Table.read(path + '/Input_data_' + str(case) + '.fits')
    logging.info('Read in table from %s /table.fits' %path)
    image_file = path + '/Grid_case' + str(case) + '.fits'
    gal_image = galsim.fits.read(image_file)
    stamp_x_size = table.meta['STAMP_X']
    stamp_y_size = table.meta['STAMP_Y']
    tab = []
    tab2 = []
    tab3 = []
    tab4 = []
    tab5 = []
    tab6 = []
    tab7 = []
    tab8 = []
    PSF = galsim.fits.read(path + '/PSF_' + str(case) + '.fits')
    meas_on_psf = ksb_moments(PSF.array, sigw = sigw_sub)
    logging.info('start ksb algorithm')
    count = 0
    positions = [((stamp_x_size*5)/2., (stamp_y_size*5)/2.)]
    aperture = CircularAperture(positions, r=sigw_sub)
    for Galaxy in table:        
        if count%100==0:
            logging.warning('Galaxy number %i' %count)
            print(str(count) + ' Galaxies examined')
        b = galsim.BoundsI(Galaxy['bound_x_left'], Galaxy['bound_x_right'], Galaxy['bound_y_bottom'], Galaxy['bound_y_top'])
        sub_gal_image = gal_image[b] 
        sub_gal_image = sub_gal_image.subsample(nx = 5,ny = 5)
        meas_on_galaxy = ksb_moments(sub_gal_image.array, sigw = sigw_sub)
        e1_anisotropy_correction = (meas_on_galaxy['Psm11'] * (meas_on_psf['e1']/meas_on_psf['Psm11']))
        #e2_anisotropy_correction = (meas_on_galaxy['Psm22'] * (meas_on_psf['e2']/meas_on_psf['Psm22']))
        P_g11      = meas_on_galaxy['Psh11']-meas_on_galaxy['Psm11']/meas_on_psf['Psm11']*meas_on_psf['Psh11']
        #P_g22 = meas_on_galaxy['Psh22']-meas_on_galaxy['Psm22']/meas_on_psf['Psm22']*meas_on_psf['Psh22']
        phot_table = aperture_photometry(sub_gal_image.array, aperture)
        cat = data_properties(sub_gal_image.array)
        columns = ['label', 'xcentroid', 'ycentroid', 'semimajor_sigma','semiminor_sigma', 'orientation']
        tbl = cat.to_table(columns=columns)
        a = tbl['semimajor_sigma'] 
        b = tbl['semiminor_sigma'] 
        my_moments = galsim.hsm.FindAdaptiveMom(sub_gal_image, guess_sig = 10, strict = (False))
        Radius = np.sqrt( a * b )
        tab.append(Radius)
        tab2.append(P_g11)
        tab3.append(e1_anisotropy_correction)
        tab4.append(meas_on_galaxy['e1'])
        tab5.append(phot_table['aperture_sum'])
        if phot_table['aperture_sum'] > 600:
            tab6.append(0)
        else:
            tab6.append(1)
        tab7.append(my_moments.moments_sigma)
        tab8.append(my_moments.moments_rho4)
        count = count + 1 
    Col_A = Column(name = 'Pg_11', data = tab2)
    Col_B = Column(name = 'anisotropy_corr', data = tab3)
    Col_C = Column(name = 'e1_cal', data = tab4)
    Col_D = Column(name = 'aperture_sum', data = tab5)
    Col_E = Column(name = 'radius_est', data = tab)
    Col_F = Column(name = 'Flag', data = tab6)
    Col_G = Column(name = 'sigma_mom', data = tab7)
    Col_H = Column(name = 'rho4_mom', data = tab8)
    try:
        table.add_columns([Col_A, Col_B, Col_C, Col_D, Col_E, Col_F, Col_G, Col_H])
        logging.info('Add columns to table')
    except:
        table.replace_column(name = 'Pg_11', col = Col_A)
        table.replace_column(name = 'anisotropy_corr', col = Col_B)
        table.replace_column(name = 'e1_cal', col = Col_C)
        table.replace_column(name = 'aperture_sum', col = Col_D)
        table.replace_column(name = 'radius_est', col = Col_E)
        table.replace_column(name = 'Flag', col = Col_F)
        table.replace_column(name = 'sigma_mom', col = Col_G)
        table.replace_column(name = 'rho4_mom', col = Col_H)
        logging.info('replaced columns in table')
    table.write( path + '/Measured_ksb_' + str(case) + '.fits' , overwrite=True) 
    logging.info('overwritten old table with new table including e1_cal and Pg_11')
    return None


def calculate_ksb_galsim(path):
    PSF = galsim.fits.read(path + '/PSF_5.fits')
    log_format = '%(asctime)s %(filename)s: %(message)s'
    logging.basicConfig(filename='ksb_galsim.log', format=log_format, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    # table = Table.read(path + '/table.fits')
    table = Table.read(path + '/Input_data_5.fits')
    logging.info('Read in table from %s /table.fits' %path)
    image_file = path + '/Grid_case5.fits'
    gal_image = galsim.fits.read(image_file)
    # ny = table.meta['NY_TILES']
    # nx = table.meta['NX_TILES']
    tab5 = []
    tab2 = []
    tab3 = []
    tab4 = []
    logging.info('start ksb algorithm')
    count = 0
    n_fail = 0
    for Galaxy in table:        
        if count%1000==0:
            logging.warning('Galaxy number %i' %count)
            print(str(count) + ' Galaxies examined')
        b = galsim.BoundsI(Galaxy['bound_x_left'], Galaxy['bound_x_right'], Galaxy['bound_y_bottom'], Galaxy['bound_y_top'])
        sub_gal_image = gal_image[b] 
        sub_gal_image = sub_gal_image.subsample(nx = 5,ny = 5)
        try:
            results = galsim.hsm.EstimateShear(sub_gal_image, PSF, guess_sig_PSF = 10, guess_sig_gal = 10)
            tab5.append(results.corrected_e1)
            tab2.append(results.observed_shape._g.real)
            tab3.append(results.observed_shape._g.imag)
            tab4.append(0)
        except:
            n_fail += 1
            tab5.append(-10)
            tab2.append(-10)
            tab3.append(-10)
            tab4.append(1)
        count = count + 1 
    Col_A = Column(name='g1_cal_galsim', data=tab2)
    Col_B = Column(name='g2_cal_galsim', data=tab3)
    Col_C = Column(name='Cal_succesful', data=tab4)
    print(n_fail)
    try:
        table.add_columns([Col_A, Col_B, Col_C])
        logging.info('Add columns to table')
    except:
        table.replace_column(name = 'g1_cal_galsim', col = Col_A)
        table.replace_column(name = 'g2_cal_galsim', col = Col_B)
        table.replace_column(name = 'Cal_succesful', col = Col_C)
        logging.info('replaced columns in table')
    table.write( path + '/Measured_galsim.fits' , overwrite=True) 
    logging.info('overwritten old table with new table including e1 and e2')
    return None
# mydir = config.workpath('Test')
# calculate_ksb_galsim(mydir)

# mydir = config.workpath('Run1/PSF_es_1')
# calculate_ksb(mydir)
# calculate_ksb_galsim('Test')