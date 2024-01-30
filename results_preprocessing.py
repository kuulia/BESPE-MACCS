#Author: Linus Lind Jan. 2024
#LICENSED UNDER: Creative Commons Attribution-ShareAlike 4.0 International
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path 

def main():
    
    def results(descriptor, target, seed, folder):
        filepath = path.relpath(f'data/KRR_output/{folder}')
        input_file = path.join(filepath,\
                                f'output_KRR_{descriptor}_{target}_{seed}.txt')
        read_file = open(input_file, 'r')
        lines = read_file.readlines()
        #input file row indexes:
        train_mae_lines = np.array([12, 30, 48, 66, 84, 102])
        train_r2_lines = train_mae_lines + 1
        train_mse_lines = train_mae_lines + 2

        test_mae_lines = train_mae_lines + 5
        test_r2_lines = test_mae_lines + 1 
        test_mse_lines = test_mae_lines + 2

        training_size_lines = train_mae_lines - 3 

        def lines_to_list(idx, lines, attr: str):
            out = []
            for line in idx:
                out.append(float(lines[line]\
                                .removeprefix(f'{attr}: ')\
                                .removesuffix('\n')))
            return out
        data = pd.DataFrame()
        data['Train_sizes'] = lines_to_list(training_size_lines, lines, \
                                            'Training_size')
        data['Train_sizes'] = data['Train_sizes'].astype(int)
        data['Train_MAE'] = lines_to_list(train_mae_lines, lines, 'Training MAE')
        data['Train_R2'] = lines_to_list(train_r2_lines, lines, 'Training r2')
        data['Train_MSE'] = lines_to_list(train_mse_lines, lines, 'Training MSE')
        data['Test_MAE'] = lines_to_list(test_mae_lines, lines, 'Test MAE')
        data['Test_R2'] = lines_to_list(test_r2_lines, lines, 'Test r2')
        data['Test_MSE'] = lines_to_list(test_mse_lines, lines, 'Test MSE')
        data.to_csv(path.join(filepath + '/results/', \
                            f'result_{descriptor}_{target}_{seed}.csv'),\
                                index=None)

    def calc_mean(descriptor, target, folder):
        random_state = [12,432,5,7543,12343,452,325432435,326,436,2435]
        filepath = path.relpath(f'data/KRR_output/{folder}/results')
        output = pd.DataFrame(np.zeros([6,7]))
        output.columns = ['Train_sizes', 'Train_MAE', 'Train_R2', 'Train_MSE', \
                        'Test_MAE', 'Test_R2', 'Test_MSE']
        for state in random_state:
            input_file = path.join(filepath,\
                                    f'result_{descriptor}_{target}_{state}.csv')
            data = pd.read_csv(input_file)
            output = output + data
        output = output / len(random_state)
        output.to_csv(path.join(filepath, f'mean_{descriptor}_{target}.csv'), \
                    index=None)
        return output

    calc_mean('MACCS_with_simpol', 'log_p_sat', 'maccs and simple simpol')
    random_state = [12,432,5,7543,12343,452,325432435,326,436,2435]
    targets = ['log_p_sat', 'kwg', 'kwiomg']
    folders = ['simpol with encodings', 'maccs and simple simpol', \
            'maccs and simpol with multiple groups', \
            'maccs and simpol with multiple groups and carbon numbers', \
            'maccs and norings simpol with binary encodings', \
            'maccs and norings simpol with binary encodings and four plus groups', \
            'maccs and simpol final model']
    for folder in folders:
        for target in targets:
            for state in random_state:
                results('MACCS_with_simpol', target, state, folder)
            calc_mean('MACCS_with_simpol', target, folder)

    for target in targets:
        for state in random_state:
            results('MACCS', target, state, 'maccs only')
        calc_mean('MACCS', target, 'maccs only')

    for target in targets:
        for state in random_state:
            results('simpol', target, state, 'simpol only')
        calc_mean('simpol', target, 'simpol only')

    #geckoq
    for folder in folders:
        if (folder != 'maccs and simple simpol' \
            and folder != 'maccs and simpol with multiple groups'):
            for state in random_state:
                results('MACCS_with_simpol_geckoq', 'log_p_sat', state, folder)
            calc_mean('MACCS_with_simpol_geckoq', 'log_p_sat', folder)

    for state in random_state:
        results('MACCS_geckoq', 'log_p_sat', state, 'maccs only')
    calc_mean('MACCS_geckoq', 'log_p_sat', 'maccs only')

    for state in random_state:
        results('simpol_fp_geckoq', 'log_p_sat', state, 'simpol only')
    calc_mean('simpol_fp_geckoq', 'log_p_sat', 'simpol only')

    #plotting
    for folder in folders:
        for target in targets:
            filepath = path.relpath(f'data/KRR_output/{folder}/results')
            input_file = path.join(filepath,\
                                f'mean_MACCS_with_simpol_{target}.csv')
            data = pd.read_csv(input_file)
            # Plot learning curve
            fig, ax = plt.subplots()
            ax.plot(data['Train_sizes'], data['Test_MAE'], marker='o')
            plt.title('Learning Curve', fontsize=18)
            ax.set_xlabel('Train Size', fontsize=18)
            ax.set_ylabel('MAE', fontsize=18)
            fig.savefig(f'{filepath}/plots/plot_learn_curve_MACCS_with_simpol_{target}.png')
            plt.close()

    for target in targets:
        # Plot learning curve       
        fig, ax = plt.subplots()
        data = pd.read_csv(f'data/KRR_output/maccs only/results/mean_MACCS_{target}.csv')
        ax.plot(data['Train_sizes'], data['Test_MAE'], marker='o')
        for folder in folders[0:5]:
            filepath = path.relpath(f'data/KRR_output/{folder}/results')
            input_file = path.join(filepath,\
                                f'mean_MACCS_with_simpol_{target}.csv')
            data = pd.read_csv(input_file)
            ax.plot(data['Train_sizes'], data['Test_MAE'], marker='o')
        legends = ['MACCS fingerprint', 'SIMPOL fingerprint', 'MACCS & SIMPOL (1)', \
                'MACCS & SIMPOL (2)', 'MACCS & SIMPOL (3)', 'MACCS & SIMPOL (4)']
        ax.legend(legends)  
        ax.set_xlabel('Train Size', fontsize=18)
        ax.set_ylabel('MAE', fontsize=18)
        plt.title('Learning Curve', fontsize=18)
        plt.close()
        outpath = path.relpath(f'data/plots/final')
        fig.savefig(f'{outpath}/plot_learn_curves_1-5_with_simpol_{target}.png')
    for target in targets:
        # Plot learning curve       
        fig, ax = plt.subplots()
        for folder in folders[4:]:
            filepath = path.relpath(f'data/KRR_output/{folder}/results')
            input_file = path.join(filepath,\
                                f'mean_MACCS_with_simpol_{target}.csv')
            data = pd.read_csv(input_file)
            ax.plot(data['Train_sizes'], data['Test_MAE'], marker='o')
        legends = ['MACCS & SIMPOL (4)', 'MACCS & SIMPOL (5)', 'MACCS & SIMPOL (6)']
        ax.legend(legends)  
        ax.set_xlabel('Train Size', fontsize=18)
        ax.set_ylabel('MAE', fontsize=18)
        plt.title('Learning Curve', fontsize=18)
        plt.close()
        outpath = path.relpath(f'data/plots/final')
        fig.savefig(f'{outpath}/plot_learn_curves_5-8_with_simpol_{target}.png')


    #geckoq
        folders_geckoq = [
            'simpol with encodings',\
            'maccs and simpol with multiple groups and carbon numbers', \
            'maccs and norings simpol with binary encodings', \
            'maccs and norings simpol with binary encodings and four plus groups', \
            'maccs and simpol final model']
    # Plot learning curve       
    fig, ax = plt.subplots()
    data = pd.read_csv(f'data/KRR_output/maccs only/results/mean_MACCS_geckoq_log_p_sat.csv')
    ax.plot(data['Train_sizes'], data['Test_MAE'], marker='o')
    for folder in folders_geckoq:
        filepath = path.relpath(f'data/KRR_output/{folder}/results')
        input_file = path.join(filepath,\
                            f'mean_MACCS_with_simpol_geckoq_log_p_sat.csv')
        data = pd.read_csv(input_file)
        ax.plot(data['Train_sizes'], data['Test_MAE'], marker='o')
    legends = ['MACCS fingerprint', 'SIMPOL fingerprint', 'MACCS & SIMPOL (3)',
            'MACCS & SIMPOL (4)', 'MACCS & SIMPOL (5)', 'MACCS & SIMPOL (6)']
    ax.legend(legends)  
    ax.set_xlabel('Train Size', fontsize=18)
    ax.set_ylabel('MAE', fontsize=18)
    plt.title('Learning Curve', fontsize=18)
    plt.close()
    outpath = path.relpath(f'data/plots/final')
    fig.savefig(f'{outpath}/plot_geckoq_learn_curves_1-4_with_simpol_log_p_sat.png')
    # Plot learning curve       
    fig, ax = plt.subplots()
    # predictions:
    wang_data = pd.read_csv('data/wang_data.csv', index_col='SMILES')
    targets = pd.DataFrame()
    preds = pd.DataFrame()
    for seed in random_state:
        filepath = path.relpath(f'data/KRR_output/maccs and simpol final model')
        input_file = path.join(filepath,\
                f'output_predictions_MACCS_with_simpol_log_p_sat_{seed}.csv')
        data = pd.read_csv(input_file)
        preds[f'preds{seed}'] = data['predictions']
        targets[f'targets{seed}'] = data['target_values']
    for col in targets.columns:
        vals = list(targets[col])
        idx = []
        for val in vals:
            idx.append(wang_data.index[wang_data['log_p_sat'] == val].tolist())
    print(idx)
if __name__ == "__main__":
    main()