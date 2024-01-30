#Author: Linus Lind Jan. 2024
#LICENSED UNDER: Creative Commons Attribution-ShareAlike 4.0 International
import pandas as pd
import numpy as np 
from os import path

def main():
    targets = ['log_p_sat', 'kwg', 'kwiomg']
    folders = ['maccs only', \
        'simpol with encodings', 'maccs and simple simpol', \
        'maccs and simpol with multiple groups', \
        'maccs and simpol with multiple groups and carbon numbers', \
        'maccs and norings simpol with binary encodings', \
        'maccs and norings simpol with binary encodings and four plus groups', \
        'maccs and simpol final model']
    for folder in folders:
        filepath = path.relpath(f'data/KRR_output/{folder}/results')
        for target in targets:
            if folder == 'maccs only':
                filename = f'mean_MACCS_{target}.csv'
            else: filename = f'mean_MACCS_with_simpol_{target}.csv'
            file = path.join(filepath, filename)
            data = pd.read_csv(file)
            data = data.drop(columns=['Train_R2', 'Train_MSE', 'Test_MSE'])
            data['Train_sizes'] = data['Train_sizes'].apply(lambda x: int(x))
            data.to_latex(F'{filepath}/{target}.tex', index=False, \
                            float_format="%.4f")
    
    


if __name__ == "__main__":
    main()