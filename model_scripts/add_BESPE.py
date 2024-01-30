#Author: Linus Lind Jan. 2024
#LICENSED UNDER: Creative Commons Attribution-ShareAlike 4.0 International
#Modifies MACCS fingerprint to create MACCS & SIMPOL (6) descriptor
import numpy as np
import pandas as pd
from os import path

def one_or_more(df):
    fingerprint = pd.DataFrame()
    simpol_groups = df.columns
    for group in simpol_groups:
        if (group != 'carbon number' and group != 'oxygen count'):
            fingerprint[group] = np.where(df[group] >= 1, 1, 0)
    return fingerprint

def two_to_four(df):
    simpol_groups = df.columns
    two_to_four_groups = pd.DataFrame()
    for group in simpol_groups:
        if (group != 'carbon number' and group != 'oxygen count'):
            two_to_four_groups[f'({group})_2-4'] = np.where((df[group] >= 2)\
                                                  & (df[group] <= 4), 1, 0)
    two_to_four_groups = two_to_four_groups.\
            loc[:, (two_to_four_groups != 0).any(axis=0)] #remove zero-columns
    return two_to_four_groups

def more_than_four(df):
    simpol_groups = df.columns
    output = pd.DataFrame()
    for group in simpol_groups:
        if (group != 'carbon number' and group != 'oxygen count'):
            output[f'({group})_4plus'] = np.where((df[group] > 4), 1, 0)
    output = output.loc[:, (output != 0).any(axis=0)]
    return output

def binary_encoded(df, element: str, name: str):
    bin_enc = pd.DataFrame()
    for i in range(0,5):
        bin_enc[f'{name}_bit{i+1}'] = df[element].apply(np.binary_repr, width = 5)\
            .map(lambda v: v[i])
    return bin_enc

def main():

    filepath = path.relpath("data")
    name_of_file = 'MACCS.txt'
    filename = path.join(filepath, name_of_file)
    maccs_fp = pd.read_csv(filename, header=None, sep=' ')
    # load SMARTS patterns
    smarts = pd.read_csv(path.join(filepath, 'enumerated_smarts_patterns.csv'))
    bespe_one_plus = one_or_more(smarts)
    bespe_two_to_four = two_to_four(smarts)
    carbons = binary_encoded(smarts, 'carbon number', 'C')
    oxygens = binary_encoded(smarts, 'oxygen count', 'O')
    bespe_four_plus = more_than_four(smarts)
    #print(bespe_two_to_four)
    for new_group in bespe_two_to_four.columns:
        groups.append(new_group)
    for carbon in carbons.columns:
        groups.append(carbon)
    for oxygen in oxygens.columns:
        groups.append(oxygen)
    for new_group in bespe_four_plus.columns:
        groups.append(new_group)
    groups.remove('carbon number')
    groups.remove('oxygen count')
    #replace unused MACCS keys with simpol fingerprints
    #print(groups)

    bespe = bespe_one_plus.join(bespe_two_to_four)
    bespe = bespe.join(carbons)
    bespe = bespe.join(oxygens)
    bespe = bespe.join(bespe_four_plus)
    print(bespe)
    maccs_and_bespe = maccs_fp.join(bespe)

    fileoutname =  f'data/MACCS_with_BESPE.txt'
    np.savetxt(fileoutname, maccs_and_bespe, fmt = "%s")

if __name__ == "__main__":
    main()