# Author: Linus Lind Jan. 2024
# LICENSED UNDER: Creative Commons Attribution-ShareAlike 4.0 International
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy
from os import path

# Function applying ideal gas law to convert between saturation vapour pressure
# p_sat to saturation mass concentration C*. 
# Units: p_sat -> log10(kPa), molar_mass -> g / mol, temp -> K
def log_psat_to_c(p_sat: float|int, \
                  molar_mass: float|int, \
                  temp: float|int) -> float: 
        R = scipy.constants.gas_constant # 8.3... Pa m^3 / (K mol)
        P = 10**(p_sat) * 1_000 # log10(kPa) -> Pa
        T = temp
        M = molar_mass
        C = (P * M) / (T * R)
        return C # units g / m^3 

def c_to_volatility_group(c: float|int) -> str:
    volatility_groups = ['ELVOC', 'LVOC', 'SVOC', 'IVOC', 'VOC']
    vol_donahue = np.array([3 * 10**(-4), 0.3, 300, 3 * 10**(6)]) # micro g / m^3
    vol_donahue = vol_donahue / 1_000_000 # unit conversion to g / m^3
    # construct list of tuples that corresponds to edges
    volatility_ranges = []
    for i, _ in enumerate(vol_donahue):
        if i < (len(vol_donahue) - 1):
            volatility_ranges.append((vol_donahue[i], vol_donahue[i+1]))
    if c <= vol_donahue[0]:
        return volatility_groups[0]
    if c >= vol_donahue[3]:
        return volatility_groups[4]
    i = 0
    max_range = volatility_ranges[i][1]
    while c > max_range:
        i += 1
        max_range = volatility_ranges[i][1]
    return volatility_groups[i+1]

def main():
    filepath = path.relpath(f'data')
    wang_data = pd.read_csv(f'{filepath}/all_smiles_molar_mass.csv')
    temp_wang = 288.15 # K
    wang_data['c'] = wang_data.apply(lambda x: log_psat_to_c(x['log_p_sat'], x['molar_mass'], temp_wang), axis=1)
    wang_data['volatility'] = wang_data.apply(lambda x: c_to_volatility_group(x['c']), axis=1)
    wang_data.to_csv(f'{filepath}/wang_data.csv', index=None)
    ##############
    wang_data = pd.read_csv(f'{filepath}/wang_data.csv', index_col='SMILES')
    filepath_preds = path.relpath(f'data/KRR_output/maccs and simpol final model')
    random_state = [12,432,5,7543,12343,452,325432435,326,436,2435]
    pred_all = pd.DataFrame()
    for seed in random_state:
        filename_preds = f'output_predictions_MACCS_with_simpol_log_p_sat_{seed}.csv'
        pred = pd.read_csv(f'{filepath_preds}/{filename_preds}', index_col='SMILES')
        pred = pred.merge(wang_data,
                          left_on=['SMILES','target_values'], 
                          right_on = ['SMILES','log_p_sat'])
        pred['c_pred'] = pred.apply(lambda x: log_psat_to_c(x['predictions'], x['molar_mass'], temp_wang), axis=1)
        pred['volatility_pred'] = pred.apply(lambda x: c_to_volatility_group(x['c_pred']), axis=1)
        pred.columns = ['log_p_sat_pred', 'target_value', 'molar_mass', 'log_p_sat', 'sat_mass_c',
                         'volatility', 'sat_mass_c_pred', 'volatility_pred']
        reordered = ['molar_mass', 'log_p_sat', 'target_value', 'log_p_sat_pred', 'sat_mass_c',
                     'sat_mass_c_pred', 'volatility', 'volatility_pred']
        pred = pred.reindex(reordered, axis=1)
        pred_all = pd.concat([pred_all, pred], axis=0)
    pred_all = pred_all.drop_duplicates()
    pred_all = pred_all.sort_values(by=['sat_mass_c'])
    pred_all.to_csv(f'{filepath}/predictions_lumiaro_log_p_sat.csv')
    wrong_preds = pred_all.query('volatility != volatility_pred')
    volatility_groups = ['ELVOC', 'LVOC', 'SVOC', 'IVOC', 'VOC']
    for group in volatility_groups:
        count_wrong_preds = len(wrong_preds[wrong_preds['volatility']==group])
        count_total_preds = len(pred_all[pred_all['volatility']==group])
        def div(x, y):
            div = 0
            if y != 0: div = x / y
            return div
        print(f'__________{group}__________\n', \
              f'Wrong predictions: {count_wrong_preds}\n', \
              f'Total predictions {count_total_preds}\n', \
              f'fraction of wrong predictions {div(count_wrong_preds, count_total_preds)}\n',\
              '__________________________________________\n')
    wrong_preds.to_csv(f'{filepath}/wrong_predictions_lumiaro_log_p_sat.csv')

    vol_donahue = np.log10(np.array([3 * 10**(-4), 0.3, 300, 3 * 10**(6)]) / 1_000_000)

    plt_pred  = np.log10(pred_all['sat_mass_c_pred'].values)
    plt_real = np.log10(pred_all['sat_mass_c'].values)
    fig, ax = plt.subplots()
    colors = ['purple', 'magenta', 'crimson', 'brown']
    for i, vol in enumerate(vol_donahue[1:]):
        ax.plot(vol * np.ones(100), np.linspace(-10,10,num=100), color='black')
        ax.fill_between(np.linspace(vol_donahue[i], vol_donahue[i+1], num=100),
                    np.ones(100) * 2* plt_real.min(), np.ones(100) * 2 *plt_real.max(), alpha=0.4, color=colors[i])
    ax.fill_between(np.linspace(vol_donahue[3], 2* plt_real.max(), num=100),
        np.ones(100) * 2* plt_real.min(), np.ones(100) * 2 *plt_real.max(), alpha=0.4, color=colors[3])
        #ax.plot(np.linspace(-10,5,num=100), vol * np.ones(100))
    ax.scatter(plt_real, plt_pred, s=5, color='navy')
    #ax.plot([plt_real.min()-5, plt_real.max()+5], [plt_real.min()-5, plt_real.max()+5], 'k--', lw=1)
    ax.plot(np.linspace(-10, 10), np.linspace(-10, 10), 'k--', lw=1)
    plt.ylim([plt_real.min()+0.1, plt_real.max()+2])
    plt.xlim([vol_donahue[0], plt_real.max()+0.1])
    #plt.title('Predicted vs. True', fontsize=14)
    ax.set_xlabel('Reference $\log_{10}(C)$', fontsize=14)  
    ax.set_ylabel('Predicted $\log_{10}(C)$', fontsize=14)
    fig.savefig(f'data/plots/maccs_with_simpol/scatter_donahue_classes.png')

if __name__ == "__main__":
    main()