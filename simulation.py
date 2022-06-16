import galsim
import numpy as np
from multiprocessing import Pool, cpu_count
import shutil
import tab
import image
import logging
import time

start = time.time()

def generate_simulation(only_one_table,path_table, path):
    log_format = '%(asctime)s %(filename)s: %(message)s'
    logging.basicConfig(filename='simulation.log', format=log_format, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('started simulation')
    if only_one_table == True:
        logging.info('Simulation uses only one randomly drawn table of galaxy parameters for the generated grid')
        image.generate_image(path_table, path, 0.3)
        shutil.copyfile(path_table, path + '/table.fits')
    else:
        logging.info('Simulation generates a new randomly drawn table of galaxy parameters for the grid')
        tab.generate_table(5,5,40,40, path)
        image.generate_image(path + '/table.fits', path, 0.3)
    return None

#generate_simulation(True, 'Test/table.fits', 'output1')
#image.generate_psf_image('Test/table.fits')

def main(N):
    paths = []
    only_one_table = []
    path_table = []
    final = [[0 for x in range(N)] for x in range(3)]
    for outputs in range(1,N+1):
        paths.append('output' + str(outputs))
        only_one_table.append(False)
        path_table.append('Test/table.fits')
    for i in range(N):
        final[i][2] = paths[i]
        final[i][0] = only_one_table[i]
        final[i][1] = path_table[i]
    with Pool() as pool:
        pool.starmap(generate_simulation, final)


if __name__ == '__main__':
    main(3)

end = time.time()
total_time = (end - start)/(60*60)  #run time in hours
print('The system took ', total_time ,' hours to execute the function')