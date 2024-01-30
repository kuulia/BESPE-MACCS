#Author: Linus Lind Jan. 2024
#LICENSED UNDER: Creative Commons Attribution-ShareAlike 4.0 International
from time import perf_counter_ns
from os import path
from os import getcwd
from model_scripts import *
import numpy as np

# the main for executing the whole code, Generates the chosen descriptors and
# runs the KRR script. Generates a summary of the results.

def main():
    ###############
    filepath = path.relpath('data')
    targets = ['p_sat_mmhg']
    generate_MACCS.main()
    #add_BESPE.main()

    ###########################################################################
    # run KRR model
    for target in targets:
        desc = 'MACCS'
        random_state = [12,432,5,7543,12343,452,325432435,326,436,2435]
        krr.main(desc, target, save_predictions=True, random_seeds=random_state, plotting=True)

if __name__ == "__main__":
    main()