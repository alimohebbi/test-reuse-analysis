import yaml

from comparision_plot_maker import DeltaMaker
from util import concat_config_results

with open(r'../config.yaml') as file:
    config = yaml.safe_load(file)



class ConfigDeltaMaker(DeltaMaker):

    def set_plot_properties(self):
        self.plot_properties = {'x_axis': 'config', 'box_width': 1, 'plot_width': 20}


if __name__ == '__main__':
    path = config['evaluator_results'] + 'atm/oracles_included/without_oracle_pass/'
    atm_df = concat_config_results(path)
    path = config['evaluator_results'] + 'craftdroid/oracles_included/'
    all_results_df = concat_config_results(path)
    craft_atm_df = all_results_df[all_results_df['src_app'].str.contains('a6|a7|a8')]
    maker = ConfigDeltaMaker(atm_df, craft_atm_df)
    maker.create_plot()
