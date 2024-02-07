#!/usr/bin/python3

#Author: Linus Lind Jan. 2024
#LICENSED UNDER: Creative Commons Attribution-ShareAlike 4.0 International
#Modifies MACCS fingerprint to BESPE-MACCS descriptor
import numpy as np
import pandas as pd
from os import path

# groups to exclude from 1, 2-4, >4 features
EXCLUDE = ['carbon number', 'oxygen count']

def remove_zero_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, (df != 0).any(axis=0)]

# Does the molecule contain exactly one count of this SMARTS pattern?
def one(df: pd.DataFrame) -> pd.DataFrame:
    fp = pd.DataFrame()
    smarts_patterns = df.columns
    for group in smarts_patterns:
        if (group not in EXCLUDE):
            fp[group] = np.where(df[group] == 1, 1, 0)
    fp = remove_zero_cols(fp)
    return fp

# Does the molecule contain 2, 3 or 4 counts of this SMARTS pattern?
def two_to_four(df: pd.DataFrame) -> pd.DataFrame:
    smarts_patterns = df.columns
    fp = pd.DataFrame()
    for group in smarts_patterns:
        if (group not in EXCLUDE):
            fp[f'({group})_2-4'] = np.where((df[group] >= 2)\
                                          & (df[group] <= 4), 1, 0)
    fp = remove_zero_cols(fp)
    return fp

# Does the molecule contain more than four counts of this SMARTS pattern?
def more_than_four(df: pd.DataFrame) -> pd.DataFrame:
    smarts_patterns = df.columns
    fp = pd.DataFrame()
    for group in smarts_patterns:
        if (group not in EXCLUDE):
            fp[f'({group})_4plus'] = np.where((df[group] > 4), 1, 0)
    fp = remove_zero_cols(fp)
    return fp

# Takes enumeration and converts to bits-bit representation 
# e.g. with bits=6: 5 -> 0 0 0 1 0 1
def binary_encoded(df: pd.DataFrame, \
                   element: str, \
                   name: str, \
                   bits: int) -> pd.DataFrame:
    bin_enc = pd.DataFrame()
    for i in range(0,bits+1):
        bin_enc[f'{name}_bit{bits+1-i}'] = df[element]\
                .apply(np.binary_repr, width = bits+1)\
                .map(lambda v: v[i])
    return bin_enc

def main():

    filepath = path.relpath("data")

    # load SMARTS patterns
    smarts = pd.read_csv(path.join(filepath, 
                                   'enumerated_smarts_patterns.csv'))
    smarts = smarts.drop(columns=['compound'])
    
    # generate binary encoded features
    bespe_one_plus = one(smarts)
    bespe_two_to_four = two_to_four(smarts)
    bespe_four_plus = more_than_four(smarts)
    carbons = binary_encoded(smarts, 'carbon number', 'C', 5) # 5-bit encoding 
    oxygens = binary_encoded(smarts, 'oxygen count', 'O', 5) 

    # combine features
    bespe = bespe_one_plus.join([bespe_two_to_four, carbons, 
                                 oxygens, bespe_four_plus])

    # load MACCS fingerprint file
    maccs_fp = pd.read_csv(path.join(filepath, 'MACCS.txt'), 
                           header=None, sep=' ')

    # append new features to MACCS and save output
    maccs_and_bespe = maccs_fp.join(bespe)
    fileoutname =  f'data/BESPE-MACCS.txt'
    np.savetxt(fileoutname, maccs_and_bespe, fmt = "%s")

if __name__ == "__main__":
    main()