import sys

import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

sys.path.append("../delta_calc")
from aggrigator.metrics_aggregator import Analyse, OracleStatus
from delta_calc.comparision_plot_maker import ComparisonPlotMaker, DeltaMaker


class ScenarioBoxPlotMaker(ComparisonPlotMaker):
    def normalize_delta(self, df):
        return df

    def __init__(self, atm_df, craft_atm_df):
        super().__init__(atm_df, craft_atm_df)
        self.sort_type = 'score'
        self.normalize = False

    def draw_plots(self, df):
        melt_result = self.make_data_ready_for_plot(df)
        if self.sort_type == 'category':
            order, melt_result = self.get_scenario_order_by_category(melt_result)
        else:
            order = self.get_scenario_order_by_score(melt_result)
        meanpointprops = Analyse.get_mean_props()
        plt.clf()
        plt.close()
        plt.figure(figsize=(20, 5))
        ax = sn.boxplot(data=melt_result, y='F1 Score', x='Scenario', hue='Test Generator', order=order, showmeans=True,
                        meanprops=meanpointprops)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        save_path = 'plot/box_plot_score_sort.pdf' if self.sort_type == 'score' else 'plot/box_plot_category_sort.pdf'
        plt.savefig(save_path, bbox_inches='tight')

    def get_scenario_order_by_score(self, melt_result):
        order = melt_result.groupby(by=["Scenario"])["F1 Score"].mean().sort_values(ascending=True).index
        return order

    def get_scenario_order_by_category(self, melt_result):
        order = melt_result.groupby(by=["Scenario"])["category"].first().sort_values(ascending=True).index
        return order, melt_result

    def make_data_ready_for_plot(self, df):
        df = df.rename(
            columns={'F1 score_atm': 'ATM', 'F1 score_craft': 'CraftDroid'})
        melt_result = pd.melt(df, id_vars=['mig_name', 'category'], value_vars=['ATM', 'CraftDroid'])
        melt_result = melt_result.rename(
            columns={'value': 'F1 Score', 'mig_name': 'Scenario', 'variable': 'Test Generator'})
        return melt_result

    def set_sort_type(self, sort_type):
        self.sort_type = sort_type


"""
    A confidence interval indicates where the population parameter is likely to reside.
    For example, a 95% confidence interval of the mean [9 11] suggests you can be 95% confident that the population
    mean is between 9 and 11.
"""


class ScenarioLinePlotMaker(ScenarioBoxPlotMaker):
    def draw_plots(self, df):
        melt_result = self.make_data_ready_for_plot(df)
        if self.sort_type == 'category':
            melt_result = self.sort_scenarios_by_category(melt_result)
        else:
            self.sort_scenarios_by_score(melt_result)
        plt.clf()
        plt.close()
        plt.figure(figsize=(20, 5))
        ax = sn.lineplot(data=melt_result, y='F1 Score', x='Scenario', hue='Test Generator')
        plt.draw()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        save_path = 'plot/line_plot_score_sort.pdf' if self.sort_type == 'score' else 'plot/line_plot_category_sort.pdf'
        plt.savefig(save_path, bbox_inches='tight')

    def sort_scenarios_by_score(self, melt_result):
        average_values = melt_result.groupby('Scenario')['F1 Score'].mean().reset_index()
        sorted_categories = average_values.sort_values(by='F1 Score')['Scenario']
        melt_result['Scenario'] = pd.Categorical(melt_result['Scenario'], categories=sorted_categories, ordered=True)

    def sort_scenarios_by_category(self, melt_result):
        categories = melt_result.groupby('Scenario')['category'].first().reset_index()
        sorted_categories = categories.sort_values(by='category')['Scenario']
        melt_result['Scenario'] = pd.Categorical(melt_result['Scenario'], categories=sorted_categories, ordered=True)
        return melt_result


class ScenarioDeltaMaker(DeltaMaker):

    def set_plot_properties(self):
        self.plot_properties = {'x_axis': 'mig_name', 'box_width': 0.5, 'plot_width': 15}


def read_result():
    analyzer = Analyse('atm', 'atm', oracles=OracleStatus.included)
    atm_result = analyzer.load_results()
    analyzer = Analyse('craftdroid', 'atm', oracles=OracleStatus.included)
    craft_results = analyzer.load_results()
    return atm_result, craft_results


if __name__ == '__main__':
    atm_result, craft_results = read_result()
    maker = ScenarioBoxPlotMaker(atm_result, craft_results)
    maker.set_sort_type('category')
    maker.create_plot()
    maker.set_sort_type('score')
    maker.create_plot()
    maker = ScenarioLinePlotMaker(atm_result, craft_results)
    maker.set_sort_type('category')
    maker.create_plot()
    maker.set_sort_type('score')
    maker.create_plot()
    maker = ScenarioDeltaMaker(atm_result, craft_results)
    maker.create_plot()
