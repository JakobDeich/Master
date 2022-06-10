import galsim
import numpy as np
from multiprocessing import Pool, cpu_count
import shutil
import tab
import image

def generate_simulation(only_one_table,path_table, path):
    if only_one_table == True:
        image.generate_image(path_table, path)
        shutil.copyfile(path_table, path + '/table.fits')
    else:
        tab.generate_table(5,5,40,40, path)
        image.generate_image(path + '/table.fits', path)
    return None

#generate_simulation(True, 'Test/table.fits', 'output1')


def main():
    paths = []
    only_one_table = []
    path_table = []
    N = 3
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
    main()
