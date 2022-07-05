import os
import sys
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.lines import Line2D
from pandas import DataFrame, Series
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import ttest_ind

with open(r'../config.yaml') as file:
    config = yaml.safe_load(file)


def save_plot(data, f, bplot):
    add_legend(f, 1)
    f.savefig("plots/" + 'impact' + ".pdf", bbox_inches='tight')


def plot_boxes_std(data, ax=None):
    f = plt.figure()
    data = pd.concat([pd.Series(v, name=k) for k, v in data.items()], axis=1)
    data = pd.DataFrame(data)
    cols = list(data.columns.values)
    print(cols)
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

    sorted_index = data.median(skipna=True).sort_values().index
    data = data[sorted_index]
    data.copy().describe().round(4).to_csv('tables/impact.csv')
    palette = get_palette()
    bplot = sns.boxplot(data=data,
                        width=0.3,
                        palette=palette, ax=ax)

    bplot = sns.stripplot(data=data,
                          jitter=True,
                          marker='o',
                          alpha=0.7,
                          color='black', size=3, ax=ax)
    bplot.set_xlabel("F1 Score", fontsize=11, fontweight='bold')
    bplot.set_ylabel("SD", fontsize=11, fontweight='bold')
    bplot.tick_params(labelsize=8)
    bplot.set(xticklabels=[])
    bplot.set_xticks([])
    if not ax:
        save_plot(data, f, bplot)


def get_index_random(data: DataFrame):
    a = set(data.query("algorithm == 'random'").index.tolist())
    return list(a)[0]


def get_values(group, t):
    desc = group.describe()
    print(desc.to_string())
    if t == "std":
        return group.std()
    else:
        return group.var()


def impact_analysis_fix_others(df: DataFrame, col: str, t: str):
    components_names = list(df.columns)
    components_names.remove(col)
    components_names.remove("value")
    group = df.groupby(components_names)["value"]
    return get_values(group, t)


def normality_test(name, data):
    k2, p = stats.normaltest(data)
    alpha = 0.05
    print("**" + str(name) + "**  ")
    # print(pd.DataFrame(data, columns=['value'])['value'].describe().to_markdown())
    alpha = 1e-3
    # print("alpha = " +str(alpha) + "  ")
    print("p-value = {:g}  ".format(p) + "  ")
    # p = 3.27207e-11
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("p <= alpha: reject H0, <span style=\"color:red\">NOT normal distribution</span><br /><br />")
    else:
        print("p > alpha: fail to reject H0, <span style=\"color:green\">normal distribution</span>  <br /><br />")


def significance_test(name1, name2, data1, data2):
    print("**" + str(name1) + " <-> " + str(name2) + "**   ")
    stat, p = ttest_ind(data1, data2)
    alpha = 0.05

    # print("alpha = 0.05  ")
    print("p-value = {:g}   ".format(p))
    # interpret
    # alpha = 0.05
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("p <= alpha: reject H0, <span style=\"color:green\">different distributions</span><br /><br />")
    else:
        print("p > alpha: fail to reject H0, <span style=\"color:red\">same distributions</span> <br /><br />")


def analyse(data: DataFrame, ax=None):
    data = remove_unrelated_rows(data)

    components_names = list(data.columns)
    components_names.remove("value")

    res_fix_others = {}
    for col in components_names:
        print("impact analysis for " + col)
        values: Series = impact_analysis_fix_others(data, col, 'std')
        print("number cluster " + str(len(values.values)))
        res_fix_others.update({col: list(values.values)})
        normality_test("std" + col, res_fix_others.get(col))

    plot_boxes_std(res_fix_others, ax=ax)
    sinigicant_test(components_names, res_fix_others)


def remove_unrelated_rows(data):
    data = data[~data['word_embedding'].isin(['random'])]
    data = data[~data['algorithm'].isin(['random'])]
    data = data[~data['training_set'].isin(['standard'])]
    data = data[~data['training_set'].isin(['empty'])]
    return data


def sinigicant_test(components_names, res_fix_others):
    combination = set()  # avoid to tst same synmmetric comb
    for col1 in components_names:
        for col2 in components_names:
            if col1 != col2 and not ((col1 + "<->" + col2) in combination):
                significance_test(col1, col2, res_fix_others.get(col1), res_fix_others.get(col2))
                combination.add(col1 + "<->" + col2)
                combination.add(col2 + "<->" + col1)


def get_palette():
    return {'training_set': 'tab:blue', 'word_embedding': 'tab:green',
            'algorithm': 'tab:orange', 'descriptors': 'orangered'}


def add_legend(fig, size=2):
    palette = get_palette()
    custom_lines = [
        Line2D([0], [0], color=j, markersize=5 * size, lw=5 * size) for j in palette.values()
    ]
    fig.legend(custom_lines, palette.keys(), title='Component type', ncol=4, loc='lower center'
               , bbox_to_anchor=(0.5, -0.1), fontsize=8 * size, title_fontsize=8 * size)


def double_plot():
    data = pd.read_csv(config['test_reuse_plot'] + '/craftdroid_all_oracle_forplot.csv')

    fig, axes = plt.subplots(1, 2, figsize=(20, 5), sharey=True)
    analyse(data, axes[0])
    data = pd.read_csv(config['test_reuse_plot'] + '/atm_atm_oracle_passfree_forplot.csv')
    analyse(data, axes[1])
    axes[0].set_title('CraftDroid', fontsize=15, fontweight='bold')
    axes[1].set_title('ATM', fontsize=15, fontweight='bold')
    add_legend(fig)
    fig.savefig('plots/impact_double.pdf', bbox_inches='tight')
    plt.show()


def single_plot(path):
    data = pd.read_csv(path)
    analyse(data)


if __name__ == "__main__":
    # double_plot()
    single_plot(config['test_reuse_plot'] + '/craftdroid_atm_oracle_excluded_forplot.csv')
