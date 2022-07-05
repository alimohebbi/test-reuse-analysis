import sys

import pandas as pd
import yaml
from matplotlib import pyplot as plt
import seaborn as sn
from matplotlib.lines import Line2D
from scipy.stats import ttest_1samp
from sklearn import preprocessing

from aggrigator.metrics_aggregator import Analyse
from util import concat_config_results, make_config_column, add_unified_mig_name
with open(r'../config.yaml') as file:
    config = yaml.safe_load(file)

def agg_atm_random_configs(atm_atm_df):
    result = atm_atm_df.groupby(by=['config', 'mig_name'], as_index=False).agg("mean")
    return result


def normalize_delta(df):
    x = df['f1_score'].values
    standard_scaler = preprocessing.StandardScaler(with_std=False)
    x_scaled = standard_scaler.fit_transform(x.reshape(-1, 1))
    df['f1_score'] = x_scaled
    return df


def config_normalization(df):
    df1_grouped = df.groupby('config')
    dfs = []
    for group_name, df_group in df1_grouped:
        df_group = normalize_delta(df_group)
        dfs.append(df_group)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = make_config_column(df)
    return df


def prepare_df_for_config_frange(df):
    df = normalize_delta(df.copy().fillna(0))
    df = make_config_column(df.fillna(0))
    df = add_unified_mig_name(df)
    df.drop(columns=['src_app', 'target_app', 'task'], inplace=True)
    return df


def config_delta_per_mig(atm_atm_df: pd.DataFrame, craft_atm_df: pd.DataFrame):
    atm_atm_df = prepare_df_for_config_frange(atm_atm_df)
    craft_atm_df = prepare_df_for_config_frange(craft_atm_df)
    atm_atm_df = agg_atm_random_configs(atm_atm_df)
    joined_dfs = pd.merge(atm_atm_df, craft_atm_df, how='inner', on=["config", "mig_name"], suffixes=("_atm", "_craft"))
    joined_dfs['delta'] = joined_dfs['F1 score_atm'] - joined_dfs['F1 score_craft']
    creat_delta_box_plots(joined_dfs)


def get_palette(data):
    palette = {}
    configs = data['config'].unique()
    for conf in configs:
        sample = data[data['config'] == conf]
        tscore, pvalue = ttest_1samp(sample['delta'], popmean=0.0)
        if pvalue / 2 < 0.05 and tscore < 0:  # one tailed ttest
            palette[conf] = 'tab:orange'
        elif pvalue / 2 < 0.05 and tscore > 0:
            palette[conf] = 'tab:blue'
        else:
            palette[conf] = 'lavender'
    return palette


def add_legend():
    custom_lines = [Line2D([0], [0], color='lavender', lw=4),
                    Line2D([0], [0], color='tab:blue', lw=4),
                    Line2D([0], [0], color='tab:orange', lw=4),
                    Line2D([0], [0], marker="^", markeredgecolor="green", markerfacecolor='red', markersize="7", lw=0)
                    ]
    plt.legend(custom_lines, ['Indifferent', 'Good for ATM', 'Good for CrafDroid', 'Mean'])


def creat_delta_box_plots(df):
    plt.clf()
    plt.close()
    plt.figure(figsize=(20, 5))
    plt.ylim((-1, 1))
    order = df.groupby(by=["config"])["delta"].mean().sort_values(ascending=True).index
    palette = get_palette(df)
    ax = sn.boxplot(data=df, y='delta', x='config', order=order, palette=palette, showmeans=True,
                    meanprops=Analyse.get_mean_props())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    add_legend()
    plt.savefig('plot/delta.pdf', bbox_inches='tight')


if __name__ == '__main__':
    path = config['evaluator_results'] + 'atm/oracles_included/without_oracle_pass/'
    atm_df = concat_config_results(path)
    path = config['evaluator_results'] + 'craftdroid/oracles_included/'
    all_results_df = concat_config_results(path)
    craft_atm_df = all_results_df[all_results_df['src_app'].str.contains('a6|a7|a8')]
    config_delta_per_mig(atm_df, craft_atm_df)
