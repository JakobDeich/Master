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
   
            
    # e1_anisotropy_correction = (meas_on_galaxy['Psm11'] * (meas_on_psf['e1']/meas_on_psf['Psm11']))
    # e2_anisotropy_correction = (meas_on_galaxy['Psm22'] * (meas_on_psf['e2']/meas_on_psf['Psm22']))
def calculate_ksb(path):
    sigw_sub = 10 
    image.generate_psf_image(path)
    PSF = galsim.fits.read(path + '/PSF.fits')
    meas_on_psf = ksb_moments(PSF.array, sigw = sigw_sub)
    log_format = '%(asctime)s %(filename)s: %(message)s'
    logging.basicConfig(filename='ksb.log', format=log_format, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    # table = Table.read(path + '/table.fits')
    table = Table.read(path + '/Input_data.fits')
    logging.info('Read in table from %s /table.fits' %path)
    image_file = path + '/Grid.fits'
    gal_image = galsim.fits.read(image_file)
    # ny = table.meta['NY_TILES']
    # nx = table.meta['NX_TILES']
    tab2 = []
    tab3 = []
    tab4 = []
    #tab5 = []
    logging.info('start ksb algorithm')
    count = 0
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
        tab2.append(P_g11)
        tab3.append(e1_anisotropy_correction)
        tab4.append(meas_on_galaxy['e1'])
        count = count + 1 
    Col_A = Column(name='Pg_11', data=tab2)
    Col_B = Column(name='anisotropy_corr', data=tab3)
    Col_C = Column(name='e1_cal', data=tab4)
    try:
        table.add_columns([Col_A, Col_B, Col_C])
        logging.info('Add columns to table')
    except:
        table.replace_column(name = 'Pg_11', col = Col_A)
        table.replace_column(name = 'anisotropy_corr', col = Col_B)
        table.replace_column(name = 'e1_cal', col = Col_C)
        logging.info('replaced columns in table')
    table.write( path + '/Measured_ksb.fits' , overwrite=True) 
    logging.info('overwritten old table with new table including e1_cal and Pg_11')
    return None

def calculate_ksb_galsim(path):
    image.generate_psf_image(path)
    PSF = galsim.fits.read(path + '/PSF.fits')
    log_format = '%(asctime)s %(filename)s: %(message)s'
    logging.basicConfig(filename='ksb_galsim.log', format=log_format, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    # table = Table.read(path + '/table.fits')
    table = Table.read(path + '/Input_data.fits')
    logging.info('Read in table from %s /table.fits' %path)
    image_file = path + '/Grid.fits'
    gal_image = galsim.fits.read(image_file)
    # ny = table.meta['NY_TILES']
    # nx = table.meta['NX_TILES']
    tab = []
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
            results = galsim.hsm.EstimateShear(sub_gal_image, PSF)
            tab.append(results.corrected_e1)
            tab2.append(results.observed_shape._g.real)
            tab3.append(results.observed_shape._g.imag)
            tab4.append(0)
        except:
            n_fail += 1
            tab.append(-10)
            tab2.append(-10)
            tab3.append(-10)
            tab4.append(1)
        count = count + 1 
    Col_A = Column(name='g1_cal_galsim', data=tab2)
    Col_B = Column(name='g2_cal_galsim', data=tab3)
    Col_C = Column(name='Cal_succesful', data=tab4)
    Col_D = Column(name= 'e1_galsim', data = tab)
    print(n_fail)
    try:
        table.add_columns([Col_A, Col_B, Col_C, Col_D])
        logging.info('Add columns to table')
    except:
        table.replace_column(name = 'g1_cal_galsim', col = Col_A)
        table.replace_column(name = 'g2_cal_galsim', col = Col_B)
        table.replace_column(name = 'Cal_succesful', col = Col_C)
        table.replace_column(name = 'e1_galsim', col = Col_D)
        logging.info('replaced columns in table')
    table.write( path + '/Measured_galsim.fits' , overwrite=True) 
    logging.info('overwritten old table with new table including e1 and e2')
    return None
# mydir = config.workpath('Test_20')
# calculate_ksb_galsim(mydir)


# calculate_ksb_galsim('Test')