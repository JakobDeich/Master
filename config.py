import os
from astropy.table import Table

workdir = "/vol/aibn1053/data1/jdeich/work_master"
gamma1 = 0.1
gamma2 = 0

gamma = Table(meta = {'gamma1': 0.1, 'gamma2': 0})
if not os.path.isdir(workdir):
    os.mkdir(workdir)
file_name = os.path.join(workdir, 'Gamma.fits')
gamma.write(file_name, overwrite = True)


 
def workpath(relpath):
    return os.path.join(workdir, relpath)
