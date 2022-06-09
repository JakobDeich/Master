import galsim
import numpy as np
from multiprocessing import Pool, cpu_count
import tab
import image

def generate_simulation(path):
    table = tab.generate_table(5,5,40,40)
    image.generate_image(table, path)
    return None

def main():
    paths = []
    N = 3
    for outputs in range(1,N+1):
        paths.append('output' + str(outputs))
    #with Pool() as pool:
    results = map(generate_simulation, paths)
    print(results)


if __name__ == '__main__':
    main()