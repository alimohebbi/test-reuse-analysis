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

from util import ReadResultCorrelationTable, ReadResultScatterPlot

ssl._create_default_https_context = ssl._create_unverified_context
with open(r'../config.yaml') as file:
    config = yaml.safe_load(file)








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
        Line2D([0], [0], marker="o", color='limegreen', markersize="6", lw=0),
        Line2D([0], [0], marker="o", color='tab:red', markersize="6", lw=0),
        Line2D([0], [0], marker="o", color='cornflowerblue', markersize="6", lw=0)
    ]
    plt.legend(custom_lines, ['Perfect', 'Random', 'Other Configs'], loc=2)


def get_palette(data):
    configs = list(data['config'])
    palette = {}
    for i in configs:
        palette[i] = 'cornflowerblue'
    palette['random_NA_NA_NA'] = 'tab:red'
    palette['perfect_NA_NA_NA'] = 'limegreen'
    return palette


def add_annotation(ax, axis_x, data):
    x_perfect = data.loc[data['Algorithm'] == 'perfect', axis_x].values[0] - 0.03
    y_perfect = data.loc[data['Algorithm'] == 'perfect', 'F1 score'].values[0] + 0.05
    ax.annotate('Perfect', (x_perfect, y_perfect))

    x_random = data.loc[data['Algorithm'] == 'random', axis_x].values[0] - 0.03
    y_random = data.loc[data['Algorithm'] == 'random', 'F1 score'].values[0] + 0.05
    ax.annotate('Random', (x_random, y_random))


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
        ax = sns.scatterplot(data=data, x=axis_x, y="F1 score", hue = 'config', palette=palette)
        add_annotation(ax, axis_x, data=data)
        m, b = np.polyfit(data[axis_x], data["F1 score"], 1)
        plt.plot(data[axis_x], m * data[axis_x] + b, color='orangered')
        add_legend()
        plt.savefig(f'plots/{axis_x}_{k.replace("_full","")}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    full_agg_results = ReadResultCorrelationTable.read_full_results()
    make_corr_tables(full_agg_results)
    full_agg_results = ReadResultScatterPlot.read_full_results()
    make_scatter_plot(full_agg_results, "MRR")
    make_scatter_plot(full_agg_results, "Top1")
