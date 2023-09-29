import numpy as np
from scipy.linalg import fractional_matrix_power
from astropy.io import fits
import scipy
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
from astropy.table import Table, Column, hstack
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


def E(x, xs1, xs2, stamp):
        A, x01, x02, M11, M12, M22 = x
        xs1 = xs1 - x01
        xs2 = xs2 - x02
        M = [[M11, M12],[M12, M22]]
        M = np.linalg.inv(M)
        hh = np.zeros((stamp.shape[0], stamp.shape[1]))
        hh = (M[0,0]*xs1+M[0,1]*xs2)*xs1+ (M[1,0]*xs1 + M[1,1]*xs2)*xs2
        #for i in range(stamp.shape[0]):
        #        xxx = []
        #        for j in range(stamp.shape[1]):
        #                xxx.append(np.matmul(np.transpose(np.array(xs)[:,i,j]),np.matmul(np.linalg.inv(M),np.array(xs)[:,i,j])))
        #        xx.append(xxx)
        #print(np.array(hh).shape)
        #print(np.matmul(np.array(xs),np.matmul(np.transpose(np.array(xs)),np.linalg.inv(M))).shape)
        return 1/2 * np.sum(np.square(stamp - A * np.exp(-1/2*np.array(hh))  ))

def Adap_w(x, y, x0, y0, M11, M12, M22):
        xs1 = x - x0  
        xs2 = y - y0  
        M = [[M11, M12],[M12, M22]]
        M = np.linalg.inv(M)
        return np.exp(-1/2*((M[0,0]*xs1+M[0,1]*xs2)*xs1+ (M[1,0]*xs1 + M[1,1]*xs2)*xs2)) 

def ksb_moments(stamp,xc=None,yc=None,sigw=2.0,prec=0.01, psf_meas = False):

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

        if psf_meas== True:
            res = scipy.optimize.minimize(E,x0 = [0.01,stamp_size/2,stamp_size/2,15,1,15], args= (x, y, stamp))
            A, x01, x02, M11, M12, M22 = res.x
            xx= np.power(x-x01,2)
            xy= np.multiply(x-x01,y-x02)
            yy= np.power(y-x02,2)
            w_adap =  Adap_w(x, y, x01, x02, 1/2*M11, 1/2*M12, 1/2*M22)
            q11_adap=np.sum(xx*w_adap*stamp)/np.sum(w_adap*stamp)
            q12_adap=np.sum(xy*w_adap*stamp)/np.sum(w_adap*stamp)
            q22_adap=np.sum(yy*w_adap*stamp)/np.sum(w_adap*stamp)
            #Q_adap = [[q11_adap, q12_adap], [q12_adap, q22_adap]]
            
            e1_adap = (q11_adap - q22_adap)/(q11_adap + q22_adap)
            e2_adap = (2*q12_adap)/(q11_adap + q22_adap)
            # print(e1_adap, e2_adap)
            
            M = [[q11_adap, q12_adap],[q12_adap, q22_adap]]
            M_inv_half_adap = fractional_matrix_power(M, (-1/2))
            # D = M11*M22-M12**2
            # zeta = D*(M11+M22+2*np.sqrt(D))
            # u_adap = 1/np.sqrt(zeta)*((M22 + np.sqrt(D))*(x-x01)+(-M12)*(y-x02))
            # v_adap = 1/np.sqrt(zeta)*((M11 + np.sqrt(D))*(y-x02)+(-M12)*(x-x01))
            u_adap = M_inv_half_adap[0][0]*(x-x01)+M_inv_half_adap[0][1]*(y-x02)
            v_adap = M_inv_half_adap[1][0]*(x-x01)+M_inv_half_adap[1][1]*(y-x02)
            # print(u_adap)
            # print(x-x01)
            
            u4_adap = np.power(u_adap,4)
            v4_adap = np.power(v_adap,4)
            u3v_adap = np.multiply(np.power(u_adap,3),v_adap)
            v3u_adap = np.multiply(np.power(v_adap,3),u_adap)

            M40_adap = np.sum(u4_adap*w_adap*stamp)/np.sum(w_adap*stamp)
            M04_adap = np.sum(v4_adap*w_adap*stamp)/np.sum(w_adap*stamp)
            M31_adap = np.sum(u3v_adap*w_adap*stamp)/np.sum(w_adap*stamp) 
            M13_adap = np.sum(v3u_adap*w_adap*stamp)/np.sum(w_adap*stamp)

            M4_1_adap = M40_adap - M04_adap
            M4_2_adap = 2*M31_adap + 2*M13_adap
            xc = x01
            yc = x02
            
            # q11_adap=np.sum(np.power(u_adap,2)*w_adap*stamp)
            # q12_adap=np.sum(np.multiply(u_adap, v_adap)*w_adap*stamp)
            # q22_adap=np.sum(np.power(v_adap,2)*w_adap*stamp)
            
            # e1_adap = (q11_adap - q22_adap)/(q11_adap + q22_adap)
            # e2_adap = (2*q12_adap)/(q11_adap + q22_adap)
            # print((q11_adap/np.sum(w_adap*stamp)*q22_adap/np.sum(w_adap*stamp)- 2*q12_adap/np.sum(w_adap*stamp))**(1/4) )
            # print('pol with adapted weight: ',e1_adap, e2_adap)

        # w = gauss2d(x, y, x01, x02, sigw, sigw)
        # xx= np.power(x-x01,2)
        # xy= np.multiply(x-x01,y-x02)
        # yy= np.power(y-x02,2)
        # print(yc, x02)

        q11=np.sum(xx*w*stamp)
        q12=np.sum(xy*w*stamp)
        q22=np.sum(yy*w*stamp)
        # Q = [[q11/np.sum(w*stamp), q12/np.sum(w*stamp)], [q12/np.sum(w*stamp), q22/np.sum(w*stamp)]]
        
        
        # M_inv_half = fractional_matrix_power(Q, (-1/2))
        # u = M_inv_half[0][0]*(x-xc)+M_inv_half[0][1]*(y-yc)
        # v = M_inv_half[1][0]*(x-xc)+M_inv_half[1][1]*(y-yc)

        # u4 = np.power(u,4)
        # v4 = np.power(v,4)
        # u3v = np.multiply(np.power(u,3),v)
        # v3u = np.multiply(np.power(v,3),u)

        # # q11_new=np.sum(np.power(u,2)*w*stamp)
        # # q12_new=np.sum(np.multiply(u, v)*w*stamp)
        # # q22_new=np.sum(np.power(v,2)*w*stamp)
        
        # # if psf_meas== True:
        # #     e1_new = (q11_new - q22_new)/(q11_new + q22_new)
        # #     e2_new = (2*q12_new)/(q11_new + q22_new)
        # #     print((q11_new/np.sum(w*stamp)*q22_new/np.sum(w*stamp)- 2*q12_new/np.sum(w*stamp))**(1/4) )
        # #     print('pol with circular weight: ', e1_new, e2_new)
        
        # M40 = np.sum(u4*w*stamp)/np.sum(w*stamp)
        # M04 = np.sum(v4*w*stamp)/np.sum(w*stamp)
        # M31 = np.sum(u3v*w*stamp)/np.sum(w*stamp) 
        # M13 = np.sum(v3u*w*stamp)/np.sum(w*stamp)
        
        
        # M4_1 = M40 - M04
        # M4_2 = 2*M31 + 2*M13

        denom= q11 + q22

        if denom!=0.0:

                e1=(q11 - q22) / denom
                e2=(2. * q12) / denom
                # if psf_meas == True:
                #     print(e1, e2)

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
                if psf_meas==True:
                    # ksbpar['M4_1']=M4_1
                    # ksbpar['M4_2']=M4_2
                    ksbpar['e1_adap']=e1_adap
                    ksbpar['e2_adap']=e2_adap
                    ksbpar['M4_1_adap']=M4_1_adap
                    ksbpar['M4_2_adap']=M4_2_adap
                return ksbpar
   
# path = config.workpath('Test')
# table = Table.read(path + '/Input_data_8.fits')
# PSF = galsim.fits.read(path + '/PSF_8.fits')
# meas_on_psf = ksb_moments(PSF.array, sigw = 10)
# print(table['psf_pol'])
# image_file = path + '/Grid_case8.fits'
# gal_image = galsim.fits.read(image_file)
# a = []
# c = []
# for Galaxy in table:
#     if Galaxy['pixel_noise']==1:
#         b = galsim.BoundsI(Galaxy['bound_x_left'], Galaxy['bound_x_right'], Galaxy['bound_y_bottom'], Galaxy['bound_y_top'])
#         sub_gal_image = gal_image[b] 
#         sub_gal_image = sub_gal_image.subsample(nx = 5,ny = 5)
#         meas_on_galaxy = ksb_moments(sub_gal_image.array, sigw = 10)
#         e1_anisotropy_correction = (meas_on_galaxy['Psm11'] * (meas_on_psf['e1']/meas_on_psf['Psm11']))
#         a.append(meas_on_galaxy['e1'])
    # else:
    #     b = galsim.BoundsI(Galaxy['bound_x_left'], Galaxy['bound_x_right'], Galaxy['bound_y_bottom'], Galaxy['bound_y_top'])
    #     sub_gal_image = gal_image[b] 
    #     sub_gal_image = sub_gal_image.subsample(nx = 5,ny = 5)
    #     meas_on_galaxy = ksb_moments(sub_gal_image.array, sigw = 10)
    #     e1_anisotropy_correction = (meas_on_galaxy['Psm11'] * (meas_on_psf['e1']/meas_on_psf['Psm11']))
    #     c.append(meas_on_galaxy['e1'])
# print(len(a), len(c))        
    # e1_anisotropy_correction = (meas_on_galaxy['Psm11'] * (meas_on_psf['e1']/meas_on_psf['Psm11']))
    # e2_anisotropy_correction = (meas_on_galaxy['Psm22'] * (meas_on_psf['e2']/meas_on_psf['Psm22']))
def calculate_ksb(path):
    sigw_sub = 10 
    log_format = '%(asctime)s %(filename)s: %(message)s'
    logging.basicConfig(filename='ksb.log', format=log_format, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    table = Table.read(path + '/Input_data.fits')
    logging.info('Read in table from %s /table.fits' %path)
    image_file = path + '/Grid.fits'
    psf_image_file = path + '/PSFs.fits'
    gal_image = galsim.fits.read(image_file)
    psf_image = galsim.fits.read(psf_image_file)
    stamp_x_size = table.meta['STAMP_X']
    stamp_y_size = table.meta['STAMP_Y']
    tab1  = []
    tab2  = []
    tab3  = []
    tab4  = []
    tab5  = []
    tab6  = []
    tab7  = []
    tab8  = []
    tab9  = []
    tab10 = []
    tab11 = []
    tab12 = []
    tab13 = []
    logging.info('start ksb algorithm')
    count = 0
    positions = [((stamp_x_size*5)/2., (stamp_y_size*5)/2.)]
    aperture = CircularAperture(positions, r=sigw_sub)
    for Galaxy in table:        
        if count%1000==0:
            logging.warning('Galaxy number %i' %count)
            print(str(count) + ' Galaxies examined')
        b = galsim.BoundsI(Galaxy['bound_x_left'], Galaxy['bound_x_right'], Galaxy['bound_y_bottom'], Galaxy['bound_y_top'])
        sub_gal_image = gal_image[b] 
        sub_psf_image = psf_image[b]
        sub_gal_image = sub_gal_image.subsample(nx = 5,ny = 5)
        meas_on_psf = ksb_moments(sub_psf_image.array, sigw = sigw_sub)
        my_moments_psf = galsim.hsm.FindAdaptiveMom(sub_psf_image, guess_sig = 10, strict = (False))
        meas_on_galaxy = ksb_moments(sub_gal_image.array, sigw = sigw_sub)
        if meas_on_galaxy is None:
            e1_anisotropy_correction = -10
            P_g11 = -10
            e1 = -10
            e2 = -10
        else:
            e1_anisotropy_correction = (meas_on_galaxy['Psm11'] * (meas_on_psf['e1']/meas_on_psf['Psm11']))
            #e2_anisotropy_correction = (meas_on_galaxy['Psm22'] * (meas_on_psf['e2']/meas_on_psf['Psm22']))
            P_g11      = meas_on_galaxy['Psh11']-meas_on_galaxy['Psm11']/meas_on_psf['Psm11']*meas_on_psf['Psh11']
            e1 = meas_on_galaxy['e1']
            e2 = meas_on_galaxy['e2']
        phot_table = aperture_photometry(sub_gal_image.array, aperture)
        cat = data_properties(sub_gal_image.array)
        columns = ['label', 'xcentroid', 'ycentroid', 'semimajor_sigma','semiminor_sigma', 'orientation']
        tbl = cat.to_table(columns=columns)
        a = tbl['semimajor_sigma'] 
        b = tbl['semiminor_sigma'] 
        my_moments = galsim.hsm.FindAdaptiveMom(sub_gal_image, guess_sig = 10, strict = (False)) 
        Radius = np.sqrt( a * b )
        tab1.append(Radius)
        tab2.append(P_g11)
        tab3.append(e1_anisotropy_correction)
        tab4.append(e1)
        tab5.append(phot_table['aperture_sum'])
        tab6.append(meas_on_psf['Q111'])
        tab7.append(my_moments.moments_sigma)
        tab10.append(my_moments_psf.moments_sigma)
        tab8.append(my_moments.moments_rho4)
        tab9.append(meas_on_psf['e1'])
        tab11.append(meas_on_psf['e2'])
        tab12.append(e2)
        tab13.append(meas_on_psf['Q222'])
        count = count + 1
    Col_A = Column(name = 'Pg_11', data = tab2)
    Col_B = Column(name = 'anisotropy_corr', data = tab3)
    Col_C = Column(name = 'e1_cal', data = tab4)
    Col_D = Column(name = 'aperture_sum', data = tab5)
    Col_E = Column(name = 'radius_est', data = tab1)
    Col_F = Column(name = 'Q111_psf', data = tab6)
    Col_G = Column(name = 'sigma_mom', data = tab7)
    Col_H = Column(name = 'rho4_mom', data = tab8)
    Col_I = Column(name = 'sigma_mom_psf', data = tab10)
    Col_J = Column(name = 'e1_cal_psf', data = tab9)
    Col_K = Column(name = 'e2_cal_psf', data = tab11)
    Col_L = Column(name = 'e2_cal', data = tab12)
    Col_M = Column(name = 'Q222_psf', data = tab13)
    try:
        table.add_columns([Col_A, Col_B, Col_C, Col_D, Col_E, Col_F, Col_G, Col_H, Col_I, Col_J, Col_K, Col_L, Col_M])
        logging.info('Add columns to table')
    except:
        table.replace_column(name = 'Pg_11', col = Col_A)
        table.replace_column(name = 'anisotropy_corr', col = Col_B)
        table.replace_column(name = 'e1_cal', col = Col_C)
        table.replace_column(name = 'aperture_sum', col = Col_D)
        table.replace_column(name = 'radius_est', col = Col_E)
        table.replace_column(name = 'Q111_psf', col = Col_F)
        table.replace_column(name = 'sigma_mom', col = Col_G)
        table.replace_column(name = 'rho4_mom', col = Col_H)
        table.replace_column(name = 'sigma_mom_psf', col = Col_I)
        table.replace_column(name = 'e1_cal_psf', col = Col_J)
        table.replace_column(name = 'e2_cal_psf', col = Col_K)
        table.replace_column(name = 'e2_cal', col = Col_L)
        table.replace_column(name = 'Q222_psf', col = Col_M)
        logging.info('replaced columns in table')
    table.write( path + '/Measured_ksb.fits' , overwrite=True) 
    logging.info('overwritten old table with new table including e1_cal and Pg_11')
    return None

def calculate_ksb_training_trys(path, case, n_rea, trys):
    with galsim.utilities.single_threaded(): 
        sigw_sub = 10 
        path = config.workpath(path)
        log_format = '%(asctime)s %(filename)s: %(message)s'
        logging.basicConfig(filename='ksb.log', format=log_format, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
        table = Table.read(path + '/Input_data_' + str(case) + '_' + str(n_rea) +'.fits')
        logging.info('Read in table from %s /table.fits' %path)
        # image_file = path + '/Grid_case' + str(case) + '.fits'
        image_file = path + '/Grid_case' + str(case) + '_' + str(trys) + '_' + str(n_rea) +'.fits'
        gal_image = galsim.fits.read(image_file)
        stamp_x_size = table.meta['STAMP_X']
        stamp_y_size = table.meta['STAMP_Y']
        tab1  = []
        tab2  = []
        tab3  = []
        tab4  = []
        tab5  = []
        tab6  = []
        tab7  = []
        tab8  = []
        tab9  = []
        tab10 = []
        tab11 = []
        tab12 = []
        tab13 = []
        PSF = galsim.fits.read(path + '/PSF_' + str(case) + '.fits')
        PSF = galsim.InterpolatedImage(PSF, scale = 0.02)
        Pixel = galsim.Pixel(scale = 0.1)
        PSF = galsim.Convolve([Pixel, PSF])
        psf_image = galsim.ImageF(table.meta['PSF_X'], table.meta['PSF_Y'], scale = 0.02)
        PSF.drawImage(psf_image,method='no_pixel') 
        my_moments_psf = galsim.hsm.FindAdaptiveMom(psf_image, guess_sig = 10, strict = (False))
        meas_on_psf = ksb_moments(psf_image.array, sigw = sigw_sub)
        logging.info('start ksb algorithm')
        count = 0
        positions = [((stamp_x_size*5)/2., (stamp_y_size*5)/2.)]
        aperture = CircularAperture(positions, r=sigw_sub)
        for Galaxy in table:        
            if count%4000==0:
                logging.warning('Galaxy number %i' %count)
                print(str(count) + ' Galaxies examined in case ' + str(case))
            b = galsim.BoundsI(Galaxy['bound_x_left'], Galaxy['bound_x_right'], Galaxy['bound_y_bottom'], Galaxy['bound_y_top'])
            sub_gal_image = gal_image[b]
            sub_gal_image = sub_gal_image.subsample(nx = 5,ny = 5)
            meas_on_galaxy = ksb_moments(sub_gal_image.array, sigw = sigw_sub)
            if meas_on_galaxy is None:
                e1_anisotropy_correction = -10
                P_g11 = -10
                e1 = -10
                e2 = -10
            else:
                e1_anisotropy_correction = (meas_on_galaxy['Psm11'] * (meas_on_psf['e1']/meas_on_psf['Psm11']))
                #e2_anisotropy_correction = (meas_on_galaxy['Psm22'] * (meas_on_psf['e2']/meas_on_psf['Psm22']))
                P_g11      = meas_on_galaxy['Psh11']-meas_on_galaxy['Psm11']/meas_on_psf['Psm11']*meas_on_psf['Psh11']
                e1 = meas_on_galaxy['e1']
                e2 = meas_on_galaxy['e2']
            phot_table = aperture_photometry(sub_gal_image.array, aperture)
            cat = data_properties(sub_gal_image.array)
            columns = ['label', 'xcentroid', 'ycentroid', 'semimajor_sigma','semiminor_sigma', 'orientation']
            tbl = cat.to_table(columns=columns)
            a = tbl['semimajor_sigma'] 
            b = tbl['semiminor_sigma'] 
            my_moments = galsim.hsm.FindAdaptiveMom(sub_gal_image, guess_sig = 10, strict = (False)) 
            Radius = np.sqrt( a * b )
            tab1.append(Radius)
            tab2.append(P_g11)
            tab3.append(e1_anisotropy_correction)
            tab4.append(e1)
            tab5.append(phot_table['aperture_sum'])
            tab6.append(meas_on_psf['Q111'])
            tab7.append(my_moments.moments_sigma)
            tab10.append(my_moments_psf.moments_sigma)
            tab8.append(my_moments.moments_rho4)
            tab9.append(meas_on_psf['e1'])
            tab11.append(meas_on_psf['e2'])
            tab12.append(e2)
            tab13.append(meas_on_psf['Q222'])
            count = count + 1
        Col_A = Column(name = 'Pg_11', data = tab2)
        Col_B = Column(name = 'anisotropy_corr', data = tab3)
        Col_C = Column(name = 'e1_cal', data = tab4)
        Col_D = Column(name = 'aperture_sum', data = tab5)
        Col_E = Column(name = 'radius_est', data = tab1)
        Col_F = Column(name = 'Q111_psf', data = tab6)
        Col_G = Column(name = 'sigma_mom', data = tab7)
        Col_H = Column(name = 'rho4_mom', data = tab8)
        Col_I = Column(name = 'sigma_mom_psf', data = tab10)
        Col_J = Column(name = 'e1_cal_psf', data = tab9)
        Col_K = Column(name = 'e2_cal_psf', data = tab11)
        Col_L = Column(name = 'e2_cal', data = tab12)
        Col_M = Column(name = 'Q222_psf', data = tab13)
        try:
    	    table.add_columns([Col_A, Col_B, Col_C, Col_D, Col_E, Col_F, Col_G, Col_H, Col_I, Col_J, Col_K, Col_L, Col_M])
    	    logging.info('Add columns to table')
        except:
            table.replace_column(name = 'Pg_11', col = Col_A)
            table.replace_column(name = 'anisotropy_corr', col = Col_B)
            table.replace_column(name = 'e1_cal', col = Col_C)
            table.replace_column(name = 'aperture_sum', col = Col_D)
            table.replace_column(name = 'radius_est', col = Col_E)
            table.replace_column(name = 'Q111_psf', col = Col_F)
            table.replace_column(name = 'sigma_mom', col = Col_G)
            table.replace_column(name = 'rho4_mom', col = Col_H)
            table.replace_column(name = 'sigma_mom_psf', col = Col_I)
            table.replace_column(name = 'e1_cal_psf', col = Col_J)
            table.replace_column(name = 'e2_cal_psf', col = Col_K)
            table.replace_column(name = 'e2_cal', col = Col_L)
            table.replace_column(name = 'Q222_psf', col = Col_M)
            logging.info('replaced columns in table')
        table.write( path + '/Measured_ksb_' + str(case) + '_' + str(trys) + '_' + str(n_rea) + '.fits' , overwrite=True) 
        logging.info('overwritten old table with new table including e1_cal and Pg_11')
    return None

def calculate_ksb_training(path, case):
    with galsim.utilities.single_threaded(): 
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
        t = Table(names = ['e1_cal', 'e2_cal', 'e1_cal_psf', 'e2_cal_psf', 'e1_cal_psf_adap', 'e2_cal_psf_adap', 'anisotropy_corr', 'anisotropy_corr_2','M4_1_psf_adap', 'M4_2_psf_adap', 'aperture_sum', 'sigma_mom', 'sigma_mom_psf', 'rho4_mom', 'Signal_to_Noise'], dtype = ['f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4', 'f4'])
        PSF = galsim.fits.read(path + '/PSF_' + str(case) + '.fits')
        PSF = galsim.InterpolatedImage(PSF, scale = 0.02)
        Pixel = galsim.Pixel(scale = 0.1)
        PSF = galsim.Convolve([Pixel, PSF])
        psf_image = galsim.ImageF(table.meta['PSF_X'], table.meta['PSF_Y'], scale = 0.02)
        PSF.drawImage(psf_image,method='no_pixel') 
        my_moments_psf = galsim.hsm.FindAdaptiveMom(psf_image, guess_sig = 10, strict = (False))
        meas_on_psf = ksb_moments(psf_image.array, sigw = sigw_sub, psf_meas=True)
        logging.info('start ksb algorithm')
        count = 0
        positions = [((stamp_x_size*5)/2., (stamp_y_size*5)/2.)]
        aperture = CircularAperture(positions, r=sigw_sub)
        for Galaxy in table:        
            if count%4000==0:
                logging.warning('Galaxy number %i' %count)
                print(str(count) + ' Galaxies examined in case ' + str(case))
            b = galsim.BoundsI(Galaxy['bound_x_left'], Galaxy['bound_x_right'], Galaxy['bound_y_bottom'], Galaxy['bound_y_top'])
            sub_gal_image = gal_image[b]
            # print(sub_gal_image.array[0])
            sigma_sky = 1.4826*np.median(abs(sub_gal_image.array[0]-np.median(sub_gal_image.array[0])))
            my_moments2 = galsim.hsm.FindAdaptiveMom(sub_gal_image, guess_sig = 2, strict = (False)) 
            Gain = 3.1
            A_eff = np.pi*(3*my_moments2.moments_sigma*np.sqrt(2*np.log(2)))**2
            Signal_to_Noise = (Gain*my_moments2.moments_amp)/np.sqrt(Gain*my_moments2.moments_amp + A_eff * (Gain*sigma_sky)**2)
            sub_gal_image = sub_gal_image.subsample(nx = 5,ny = 5)
            meas_on_galaxy = ksb_moments(sub_gal_image.array, sigw = sigw_sub)
            if meas_on_galaxy is None:
                e1_anisotropy_correction = -10
                e2_anisotropy_correction = -10
                e1 = -10
                e2 = -10
            else:
                e1_anisotropy_correction = (meas_on_galaxy['Psm11'] * (meas_on_psf['e1']/meas_on_psf['Psm11']))
                e2_anisotropy_correction = (meas_on_galaxy['Psm22'] * (meas_on_psf['e2']/meas_on_psf['Psm22']))
                #P_g11      = meas_on_galaxy['Psh11']-meas_on_galaxy['Psm11']/meas_on_psf['Psm11']*meas_on_psf['Psh11']
                e1 = meas_on_galaxy['e1']
                e2 = meas_on_galaxy['e2']
            if meas_on_psf is None:
                par = ['e1', 'e2', 'e1_adap', 'e2_adap', 'M4_1_adap', 'M4_2_adap']
                for i in par:
                    meas_on_psf[par] = -10
            phot_table = aperture_photometry(sub_gal_image.array, aperture)
            # cat = data_properties(sub_gal_image.array)
            # columns = ['label', 'xcentroid', 'ycentroid', 'semimajor_sigma','semiminor_sigma', 'orientation']
            # tbl = cat.to_table(columns=columns)
            # a = tbl['semimajor_sigma'] 
            # b = tbl['semiminor_sigma'] 
            my_moments = galsim.hsm.FindAdaptiveMom(sub_gal_image, guess_sig = 10, strict = (False)) 
            # Radius = np.sqrt( a * b )
            print(my_moments.moments_sigma)
            params = [e1, e2, meas_on_psf['e1'], meas_on_psf['e2'], meas_on_psf['e1_adap'], meas_on_psf['e2_adap'], e1_anisotropy_correction, e2_anisotropy_correction, meas_on_psf['M4_1_adap'], meas_on_psf['M4_2_adap'], phot_table['aperture_sum'], my_moments.moments_sigma, my_moments_psf.moments_sigma, my_moments.moments_rho4, Signal_to_Noise]
            t.add_row(params)
            count = count + 1
        print('KSB finished')
        table_new = hstack([table, t])
        table_new.write( path + '/Measured_ksb_' + str(case) + '.fits' , overwrite=True) 
        logging.info('overwritten old table with new table including e1_cal and Pg_11')
    return None

path = config.workpath('Test5')
#t = Table.read(path + '/Measured_ksb.fits')
#print(t['M4_2_psf'], max(abs(t['M4_2_psf'])))
calculate_ksb_training(path, 6)
# calculate_ksb_training(path, 1)
# calculate_ksb_training(path, 2)
# calculate_ksb_training(path, 3)

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
