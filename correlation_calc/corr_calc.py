import glob
import os

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pandas import read_csv
from scipy.stats import pearsonr
import seaborn as sns
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
with open(r'../config.yaml') as file:
    config = yaml.safe_load(file)


def add_baseline_values(csv_df, file_name):
    random_config_mrr = {'all': 0.20191013, 'atm': 0.2018414, 'craft': 0.196925}
    random_config_top1 = {'all': 19.5 / 337, 'atm': 20.5 / 116, 'craft': 19 / 221}
    csv_df.loc[csv_df['Algorithm'] == 'perfect', 'MRR'] = 1
    csv_df.loc[csv_df['Algorithm'] == 'perfect', 'Top1'] = 1
    if '_craft' in file_name:
        csv_df.loc[csv_df['Algorithm'] == 'random', 'MRR'] = random_config_mrr['craft']
        csv_df.loc[csv_df['Algorithm'] == 'random', 'Top1'] = random_config_top1['craft']
    elif '_atm' in file_name:
        csv_df.loc[csv_df['Algorithm'] == 'random', 'MRR'] = random_config_mrr['atm']
        csv_df.loc[csv_df['Algorithm'] == 'random', 'Top1'] = random_config_top1['atm']
    else:
        csv_df.loc[csv_df['Algorithm'] == 'random', 'MRR'] = random_config_mrr['all']
        csv_df.loc[csv_df['Algorithm'] == 'random', 'Top1'] = random_config_top1['all']
    return csv_df


def read_results(plot=False):
    full_agg_results = {}
    for path in glob.glob(config['test_reuse_full'] + "/*.csv"):
        csv_df = read_csv(path, encoding='latin-1')
        file_name = os.path.basename(path).split('.')[0]
        if not plot:
            csv_df = csv_df[csv_df['MRR'].notna()]
        else:
            csv_df = add_baseline_values(csv_df, file_name)
        full_agg_results[file_name] = csv_df
    return full_agg_results


def make_corr_tables(full_agg_results):
    corr_list = []
    for k, v in full_agg_results.items():
        corr_m, p_value_m = pearsonr(list(v['F1 score']), list(v['MRR']))
        corr_t, p_value_t = pearsonr(list(v['F1 score']), list(v['Top1']))
        row = {'Name': k, 'Corr MRR': corr_m, 'p_value_m': float(p_value_m), 'Corr Top1': corr_t,
               'p_value_t': float(p_value_t)}
        corr_list.append(row)
    corr_df = pd.DataFrame(corr_list)
    corr_df.round(5).to_csv('corr.csv', index=False, float_format='%.5f')


def add_legend():
    custom_lines = [
        Line2D([0], [0], marker="o", color='limegreen', markersize="8", lw=0),
        Line2D([0], [0], marker="o", color='tab:red', markersize="7", lw=0),
        Line2D([0], [0], marker="o", color='gold', markersize="6", lw=0),
        Line2D([0], [0], marker="o", color='cornflowerblue', markersize="5", lw=0)
    ]
    plt.legend(custom_lines, ['Perfect', 'Random', 'Syntactic Configs',  'Other Configs'], loc=2)


def get_palette(data):
    configs = list(data['config'])
    data['size'] = 50
    palette = {}
    for i in configs:
        palette[i] = 'cornflowerblue'
        if 'es' in i or 'js' in i:
            palette[i] = 'gold'
            data.loc[data['config']==i,'size'] = 65
    palette['random_NA_NA_NA'] = 'tab:red'
    palette['perfect_NA_NA_NA'] = 'limegreen'
    data.loc[data['config']=='random_NA_NA_NA','size'] = 80
    data.loc[data['config']=='perfect_NA_NA_NA','size'] = 100
    return palette


def make_scatter_plot(full_agg_results, axis_x):
    for k, v in full_agg_results.items():
        data = v.copy().fillna('NA')
        data['config'] = data['Algorithm'] + '_' + data['Training'] + '_' + data['Embedding'] + '_' + data[
            'Descriptors']
        plt.clf()
        plt.close()
        plt.figure(figsize=(10, 5))
        plt.ylim(0, 1)
        palette = get_palette(data)
        ax = sns.scatterplot(data=data, x=axis_x, y="F1 score", hue='config', palette=palette, size='size', sizes=(50,200),
                             legend=False)
        m, b = np.polyfit(data[axis_x], data["F1 score"], 1)
        plt.plot(data[axis_x], m * data[axis_x] + b, color='orangered')
        add_legend()
        plt.savefig(f'plots/{axis_x}_{k}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    full_agg_results = read_results(False)
    make_corr_tables(full_agg_results)
    full_agg_results = read_results(True)
    make_scatter_plot(full_agg_results, "MRR")
    make_scatter_plot(full_agg_results, "Top1")
