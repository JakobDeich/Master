import numpy as np
from astropy.io import fits
import sys
import os
import math
import galsim
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
#import image
import tab
from astropy.table import QTable

table = tab.generate_table()
#image.generate_image()
image_file = 'output/Grid.fits'
image_file_psf = 'output/Grid_epsf.fits'

gal_image = galsim.fits.read(image_file)
psf_image = galsim.fits.read(image_file_psf)
gal_image = gal_image.subsample(nx = 5,ny = 5)

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
            
def calculate_ksb():
    tab2 = []
    tab3 = []
    for i in range(10000):
        b = table[i][1]
        sub_gal_image = gal_image[b] 
        sub_psf_image = psf_image[b]
        ksb_mom = ksb_moments(sub_gal_image.array)
        tab2.append(ksb_mom['e1'])
        tab3.append(ksb_mom['e2'])
    print(tab2)
    tab = table[i][0]
    t = QTable([tab, tab2, tab3])
    return t
Test = calculate_ksb()
