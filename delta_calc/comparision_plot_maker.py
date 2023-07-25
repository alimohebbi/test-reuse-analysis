from abc import ABC, abstractmethod

import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ttest_1samp
from sklearn import preprocessing

from aggrigator.metrics_aggregator import Analyse
from util import make_config_column, add_unified_mig_name, make_mig_name_readable


class ComparisonPlotMaker(ABC):
    def __init__(self, atm_df, craft_df):
        self.normalize = False
        self.atm_df = atm_df.copy()
        self.craft_df = craft_df.copy()

    @staticmethod
    def agg_atm_random_configs(atm_atm_df):
        result = atm_atm_df.groupby(by=['config', 'mig_name'], as_index=False).agg("mean")
        return result

    def prepare_df_for_config_frange(self, df):
        if self.normalize:
            df = self.normalize_delta(df.copy().fillna(0))
        df = make_config_column(df.fillna(0))
        df = add_unified_mig_name(df)
        df.drop(columns=['src_app', 'target_app', 'task'], inplace=True)
        return df

    def get_processed_joined_df(self):
        atm_atm_df = self.prepare_df_for_config_frange(self.atm_df)
        craft_atm_df = self.prepare_df_for_config_frange(self.craft_df)
        atm_atm_df = self.agg_atm_random_configs(atm_atm_df)
        joined_dfs = pd.merge(atm_atm_df, craft_atm_df, how='inner', on=["config", "mig_name"],
                              suffixes=("_atm", "_craft"))
        joined_dfs['delta'] = joined_dfs['F1 score_atm'] - joined_dfs['F1 score_craft']
        return joined_dfs

    @abstractmethod
    def normalize_delta(self, df):
        pass

    def create_plot(self):
        joined_dfs = self.get_processed_joined_df()
        joined_dfs = make_mig_name_readable(joined_dfs)
        self.draw_plots(joined_dfs)

    @abstractmethod
    def draw_plots(self, joined_dfs):
        pass


class DeltaMaker(ComparisonPlotMaker, ABC):
    def __init__(self, atm_df, craft_df):
        super().__init__(atm_df, craft_df)
        self.normalize = True
        self.plot_properties = None
        self.set_plot_properties()

    @abstractmethod
    def set_plot_properties(self):
        pass

    def normalize_delta(self, df):
        x = df['f1_score'].values
        standard_scaler = preprocessing.StandardScaler(with_std=False)
        x_scaled = standard_scaler.fit_transform(x.reshape(-1, 1))
        df['f1_score'] = x_scaled
        return df

    def get_palette(self, data):
        palette = {}
        configs = data[self.plot_properties['x_axis']].unique()
        for conf in configs:
            sample = data[data[self.plot_properties['x_axis']] == conf]
            tscore, pvalue = ttest_1samp(sample['delta'], popmean=0.0)
            if pvalue / 2 < 0.05 and tscore < 0:  # one tailed ttest
                palette[conf] = 'tab:orange'
            elif pvalue / 2 < 0.05 and tscore > 0:
                palette[conf] = 'tab:blue'
            else:
                palette[conf] = 'lavender'
        return palette

    @staticmethod
    def add_legend():
        custom_lines = [Line2D([0], [0], color='lavender', lw=4),
                        Line2D([0], [0], color='tab:blue', lw=4),
                        Line2D([0], [0], color='tab:orange', lw=4),
                        Line2D([0], [0], marker="^", markeredgecolor="green", markerfacecolor='red', markersize="7",
                               lw=0)
                        ]
        plt.legend(custom_lines, ['Indifferent', 'Good for ATM', 'Good for CrafDroid', 'Mean'])

    def draw_plots(self, df):
        plt.clf()
        plt.close()
        plt.figure(figsize=(self.plot_properties['plot_width'], 5))
        plt.ylim((-1, 1))
        order = df.groupby(by=[self.plot_properties['x_axis']])["delta"].mean().sort_values(ascending=True).index
        palette = self.get_palette(df)
        ax = sn.boxplot(data=df, y='delta', x=self.plot_properties['x_axis'], order=order, palette=palette,
                        showmeans=True,
                        meanprops=Analyse.get_mean_props(), width=self.plot_properties['box_width'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        DeltaMaker.add_legend()
        plt.savefig('plot/delta.pdf', bbox_inches='tight')

