import pandas as pd
from matplotlib import pyplot as plt

from aggrigator.metrics_aggregator import Analyse, OracleStatus
import seaborn as sn

from delta_calc.delta_calc import get_processed_joined_df
from util import add_mig_name, convert_config_names, make_mig_name_readable


def pre_process(df):
    df.fillna(0, inplace=True)
    df = add_mig_name(df)
    df = convert_config_names(df)
    return df


def read_result():
    analyzer = Analyse('atm', 'atm', oracles=OracleStatus.included)
    atm_result = analyzer.load_results()
    analyzer = Analyse('craftdroid', 'atm', oracles=OracleStatus.included)
    craft_results = analyzer.load_results()
    return atm_result, craft_results


def make_box_plot(df):
    df = df.copy()
    plt.clf()
    plt.close()
    plt.figure(figsize=(20, 5))
    df = df.rename(
        columns={'F1 score_atm': 'ATM', 'F1 score_craft': 'CraftDroid'})
    melt_result = pd.melt(df, id_vars=['mig_name'], value_vars=['ATM', 'CraftDroid'])
    melt_result = melt_result.rename(columns={'value': 'F1 Score', 'mig_name': 'Scenario', 'variable': 'Test Generator'})
    order = melt_result.groupby(by=["Scenario"])["F1 Score"].mean().sort_values(ascending=True).index
    meanpointprops = Analyse.get_mean_props()
    ax = sn.boxplot(data=melt_result, y='F1 Score', x='Scenario', hue='Test Generator', order=order, showmeans=True,
                    meanprops=meanpointprops)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig('box_plot.pdf', bbox_inches='tight')


def make_line_plot(df):
    df = df.copy()
    plt.clf()
    plt.close()
    plt.figure(figsize=(20, 5))
    df = df.rename(
        columns={'F1 score_atm': 'ATM', 'F1 score_craft': 'CraftDroid'})
    melt_result = pd.melt(df, id_vars=['mig_name'], value_vars=['ATM', 'CraftDroid'])
    melt_result = melt_result.rename(columns={'value': 'F1 Score', 'mig_name': 'Scenario', 'variable': 'Test Generator'})

    average_values = melt_result.groupby('Scenario')['F1 Score'].mean().reset_index()
    sorted_categories = average_values.sort_values(by='F1 Score')['Scenario']
    melt_result['Scenario'] = pd.Categorical(melt_result['Scenario'], categories=sorted_categories, ordered=True)

    ax = sn.lineplot(data=melt_result, y='F1 Score', x='Scenario', hue='Test Generator')
    plt.draw()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig('line_plot.pdf', bbox_inches='tight')


atm_result, craft_results = read_result()
joined_results = get_processed_joined_df(atm_result, craft_results, normalize=False)
joined_results = make_mig_name_readable(joined_results)
make_box_plot(joined_results)
make_line_plot(joined_results)
