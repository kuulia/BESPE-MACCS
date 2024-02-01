#!/usr/bin/python3
#Author: Linus Lind Jan. 2024
#LICENSED UNDER: Creative Commons Attribution-ShareAlike 4.0 International
from model_scripts import *

# the main for executing the whole code, Generates the chosen descriptors and
# runs the KRR script. Generates a summary of the results.

def main():

    # generate descriptors
    generate_MACCS.main()
    add_BESPE.main()

    # select descriptor and target values to use 
    # (these should correspond to file names in data folder)
    descriptor = 'MACCS_with_BESPE'
    target = 'log10_p_sat_kpa'
    random_state = [12, 432, 5, 7543, 12343, \
                    452, 325432435, 326, 436, 2435]
    # run KRR model
    krr.main(descriptor, target, save_predictions=True, \
             random_seeds=random_state, plotting=True)

if __name__ == "__main__":
    main()