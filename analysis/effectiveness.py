import glob
import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from pandas import DataFrame, read_csv

from analysis.impact_analysis import significance_test
from util import ReadResultAnalysis

with open(r'../config.yaml') as file:
    config = yaml.safe_load(file)


def write_effectiveness(data, component_to_els, dir):
    for i in component_to_els.keys():
        if i == 'training_set':
            component_to_els[i].remove('standard')
            component_to_els[i].remove('empty')
        temp = data[component_to_els[i]].copy()
        sorted_index = temp.median(skipna=True).sort_values().index
        temp = temp[sorted_index]
        temp.describe().round(4).to_csv(f'tables/{dir}/{i}_eff.csv')


def analyse(data: DataFrame, dir):
    value_column = data["value"]
    print(value_column.describe())
    print("the BEST configuration is " + os.linesep + str(data.loc[value_column.idxmax()]))
    print("the WORST configuration is " + os.linesep + str(data.loc[value_column.idxmin()]))

    components_names = list(data.columns)
    components_names.remove("value")
    positions = []
    for top in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75]:
        positions.append(top * 253)
    # build blox plots_p values
    col_index = 0
    result_string = ""
    for col in components_names:
        result_string = result_string + "\n"
        for compo in list(data[col].unique()):
            result_string = result_string + compo + ","
        result_string = result_string + "\n"
        for pos in positions:
            result_string = result_string + "\n1 to " + str(int(pos) + 1) + ","
            res = data.loc[0:int(pos), :][col].value_counts(normalize=True)
            dictionary: dict = res.to_dict()
            for compo in list(data[col].unique()):
                if compo in dictionary:
                    result_string = result_string + str(dictionary.get(compo)) + ","
                else:
                    result_string = result_string + "0.00,"
            print(data.loc[0:int(pos), :][col].value_counts(normalize=True))
        col_index = col_index + 1
    print(result_string)
    data = data[~data['word_embedding'].isin(['random'])]
    data = data[~data['algorithm'].isin(['random'])]
    # get the header of the columns
    components_names = list(data.columns)
    components_names.remove("value")
    # build blox plots_p values
    res_fix_comp_all = {}
    component_to_elements = {}
    for col in components_names:
        res_fix_comp_single = {}
        for el in list(data[col].unique()):
            if el != "empty" and el != "standard":
                res_fix_comp_single.update({el: data["value"].loc[data[col] == el]})
                res_fix_comp_all.update({el: data["value"].loc[data[col] == el]})
        component_to_elements.update({col: list(data[col].unique())})

    combination = set()  # avoid to tst same synmmetric comb
    for col in components_names:
        for el1 in list(data[col].unique()):
            if el1 != "empty" and el1 != "standard":
                for el2 in list(data[col].unique()):
                    if el2 != "empty" and el2 != "standard":
                        if el2 != el1 and not ((el1 + "<->" + el2) in combination):
                            significance_test(el1, el2, res_fix_comp_all.get(el1), res_fix_comp_all.get(el2))
                            combination.add(el1 + "<->" + el2)
                            combination.add(el2 + "<->" + el1)

        # plot_boxes(res_fix_comp_single, 'value distribution for component' + str(col), str(col), 'evaluation metric',
        #           str(col) + "-value-distribution", {})
    plot_boxes(res_fix_comp_all, 'MRR'
               , component_to_elements, dir)


def plot_boxes(data: DataFrame, y_axis_label: str,
               component_to_els: dict, dir):
    f = plt.figure()
    data = pd.concat([pd.Series(v, name=k) for k, v in data.items()], axis=1)
    data = pd.DataFrame(data)
    write_effectiveness(data, component_to_els, dir)
    cols = list(data.columns.values)
    print(cols)

    # -----------------------------------
    # HARDCODE
    # 3 options: MRR, TOP1 and MEDIAN
    # order by median but for component
    # 1) MRR
    train_sets = ['blogs', 'manuals', 'googleplay']
    others = ['js', 'bert', 'es', 'w2v', 'fast', 'glove', 'nnlm', 'use', 'wm',
              'craftdroid_d', 'union', 'intersection', 'atm_d',
              'craftdroid_a', 'atm_a', 'semfinder_a', 'adaptdroid_a']
    all_axis = []
    all_axis.extend(others)
    all_axis.extend(train_sets)
    data = data.reindex(
        all_axis, axis=1)

    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

    bplot = sns.boxplot(data=data,
                        width=0.45,
                        palette='colorblind')

    bplot = sns.stripplot(data=data,
                          jitter=True,
                          marker='o',
                          alpha=0.5,
                          color='black')
    bplot.set_ylabel(y_axis_label, fontsize=16, fontweight='bold')
    bplot.tick_params(labelsize=15)
    plt.setp(bplot.get_xticklabels(), rotation=45)

    f.set_size_inches(data.columns.size * 1.5, 4)

    # color based on component type
    handles = []
    if len(component_to_els) != 0:
        colors = sns.color_palette('colorblind')
        j = 0
        for els in component_to_els.values():
            i = 0
            for col in data.columns:
                if col in els:
                    bplot.artists[i].set_facecolor(colors[j])
                i = i + 1
            handles.append(mpatches.Patch(color=colors[j], label='Label1'))
            j = j + 1
    f.savefig(os.path.join("plots", dir, 'all_value_distribution.pdf'), bbox_inches='tight')
    plt.close('all')


if __name__ == '__main__':

    full_agg_results = ReadResultAnalysis().read_full_results()
    for dir, df in full_agg_results.items():
        dir = dir.replace('_forplot','')
        if not os.path.exists(os.path.join('tables', dir)):
            os.mkdir(os.path.join('tables', dir))
        if not os.path.exists(os.path.join('plots', dir)):
            os.mkdir(os.path.join('plots', dir))
        analyse(df, dir)
