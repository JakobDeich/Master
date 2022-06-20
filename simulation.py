
import galsim
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import shutil
import tab
import image
import logging
import time

import config

start = time.time()

def generate_simulation(only_one_table = False, path_table = 'Test/table.fits', dirname='test'):   
    mydir = config.workpath(dirname)
    mylogpath = os.path.join(mydir, "generate_simulation.log")
    log_format = '%(asctime)s %(filename)s: %(message)s'
    logging.basicConfig(filename=mylogpath, format=log_format, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('started simulation')
    if only_one_table == True:
        logging.info('Simulation uses only one randomly drawn table of galaxy parameters for the generated grid')
        image.generate_image(path_table, mydir)
        shutil.copyfile(path_table, mydir + '/table.fits')
    else:
        logging.info('Simulation generates a new randomly drawn table of galaxy parameters for the grid')
        tab.generate_table(5,5,40,40, mydir)
        image.generate_image(mydir + '/table.fits', mydir)
    return None

#generate_simulation(True, 'Test/table.fits', 'output1')
#image.generate_psf_image('Test/table.fits')

def main(N):
    runname = "output"
    final = []
    for i in range(N):
        params = [True,'Test/table.fits' ,runname+"_" + str(i+1)]
        final.append(params)
    with Pool() as pool:
        pool.starmap(generate_simulation, final)



if __name__ == '__main__':
    main(3)

end = time.time()
total_time = (end - start)/(60*60)  #run time in hours
print('The system took ', total_time ,' hours to execute the function')