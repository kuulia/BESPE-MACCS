#!/usr/bin/python3
#Author: Linus Lind Jan. 2024
#LICENSED UNDER: Creative Commons Attribution-ShareAlike 4.0 International
#Modifies MACCS fingerprint to create MACCS & SIMPOL (6) descriptor
import numpy as np
import pandas as pd
from os import path

# groups to exclude from 1+, 2-4, 4+ features
EXCLUDE = ['carbon number', 'oxygen count']

def one_or_more(df: pd.DataFrame) -> pd.DataFrame:
    fp = pd.DataFrame()
    smarts_patterns = df.columns
    for group in smarts_patterns:
        if (group not in EXCLUDE):
            fp[group] = np.where(df[group] >= 1, 1, 0)
    return fp

def two_to_four(df: pd.DataFrame) -> pd.DataFrame:
    smarts_patterns = df.columns
    fp = pd.DataFrame()
    for group in smarts_patterns:
        if (group not in EXCLUDE):
            fp[f'({group})_2-4'] = np.where((df[group] >= 2)\
                                                  & (df[group] <= 4), 1, 0)
    fp = fp.loc[:, (fp != 0).any(axis=0)] #remove zero-columns
    return fp

def more_than_four(df: pd.DataFrame) -> pd.DataFrame:
    smarts_patterns = df.columns
    fp = pd.DataFrame()
    for group in smarts_patterns:
        if (group not in EXCLUDE):
            fp[f'({group})_4plus'] = np.where((df[group] > 4), 1, 0)
    fp = fp.loc[:, (fp != 0).any(axis=0)]
    return fp

def binary_encoded(df: pd.DataFrame, element: str, name: str, bits: int) -> pd.DataFrame:
    bin_enc = pd.DataFrame()
    for i in range(0,bits+1):
        bin_enc[f'{name}_bit{bits+1-i}'] = df[element].apply(np.binary_repr, width = bits+1)\
            .map(lambda v: v[i])
    return bin_enc

def main():

    filepath = path.relpath("data")

    # load SMARTS patterns
    smarts = pd.read_csv(path.join(filepath, 'enumerated_smarts_patterns.csv'))
    smarts = smarts.drop(columns=['compound'])
    
    # generate binary encoded features
    bespe_one_plus = one_or_more(smarts)
    bespe_two_to_four = two_to_four(smarts)
    bespe_four_plus = more_than_four(smarts)
    carbons = binary_encoded(smarts, 'carbon number', 'C', 5)
    oxygens = binary_encoded(smarts, 'oxygen count', 'O', 5)

    # combine features
    bespe = bespe_one_plus.join(bespe_two_to_four)
    bespe = bespe.join(carbons)
    bespe = bespe.join(oxygens)
    bespe = bespe.join(bespe_four_plus)

    # load MACCS fingerprint file
    maccs_fp = pd.read_csv(path.join(filepath, 'MACCS.txt'), header=None, sep=' ')

    # append new features to MACCS
    maccs_and_bespe = maccs_fp.join(bespe)
    fileoutname =  f'data/MACCS_with_BESPE.txt'
    np.savetxt(fileoutname, maccs_and_bespe, fmt = "%s")

if __name__ == "__main__":
    main()