import re

import pandas as pd
import seaborn as sn
import yaml
from matplotlib import pyplot as plt

from util import rename_subjects, add_mig_name, concat_config_results

with open(r'../config.yaml') as file:
    config = yaml.safe_load(file)


def describe_f1_per_migration(all_results_df, save_path):
    groups_by = ['src_app', 'target_app']
    grouped_results = all_results_df.groupby(by=groups_by)
    group_desc = grouped_results['f1_score'].describe()
    group_desc.to_csv(save_path)


def creat_box_plots(df, column, save_path):
    plt.clf()
    plt.close()
    plt.figure(figsize=(20, 5))
    df = add_mig_name(df)
    ax = sn.boxplot(data=df, y=column, x='mig_name')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    sn.stripplot(data=df, y=column, x='mig_name', jitter=True,
                 dodge=True,
                 marker='o',
                 alpha=0.5)
    plt.savefig(save_path, bbox_inches='tight')


def get_tool_type(x):
    return 'craft' if bool(re.search('a[6-8]', x['src_app'])) else 'atm'


def creat_box_plots_sbs(df, column, save_path):
    plt.clf()
    plt.close()
    plt.figure(figsize=(20, 5))
    if 'task' not in df.columns:
        df['task'] = ''
    df['mig_name'] = df['src_app'] + ' - ' + df['target_app'] + ' - ' + df['task']
    df['tool'] = ''
    df['tool'] = df.apply(get_tool_type, axis=1)
    df['mig_name'] = df.apply(rename_subjects, axis=1)
    ax = sn.boxplot(data=df, y=column, x='mig_name', hue='tool')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    sn.stripplot(data=df, y=column, x='mig_name', hue='tool', jitter=True,
                 dodge=True,
                 marker='o',
                 alpha=0.5)
    plt.savefig(save_path, bbox_inches='tight')


if __name__ == '__main__':
    path = config['evaluator_results'] + 'atm/oracles_included/without_oracle_pass/'
    atm_df = concat_config_results(path)
    describe_f1_per_migration(atm_df, 'table/atm.csv')
    creat_box_plots(atm_df, 'f1_score', 'plots/atm_f1_range.pdf')

    path =  config['evaluator_results'] + 'craftdroid/oracles_included/'
    all_results_df = concat_config_results(path)
    describe_f1_per_migration(all_results_df, 'table/craft_all.csv')
    creat_box_plots(all_results_df, 'f1_score', 'plots/craft_all_f1_range.pdf')

    craft_craft_df = all_results_df[~all_results_df['src_app'].str.contains('a6|a7|a8')]
    describe_f1_per_migration(craft_craft_df, 'table/craft_craft.csv')
    creat_box_plots(craft_craft_df, 'f1_score', 'plots/craft_craft_f1_range.pdf')

    craft_atm_df = all_results_df[all_results_df['src_app'].str.contains('a6|a7|a8')]
    describe_f1_per_migration(craft_atm_df, 'table/craft_atm.csv')
    creat_box_plots(craft_atm_df, 'f1_score', 'plots/craft_atm_f1_range.pdf')

    side_by_side_df = pd.concat([atm_df, craft_atm_df])
    creat_box_plots_sbs(side_by_side_df, 'f1_score', 'plots/craft_sbs_f1_range.pdf')
