import glob
import os

import pandas as pd
import yaml
from pandas import read_csv
from scipy.stats import pearsonr

with open(r'../config.yaml') as file:
    config = yaml.safe_load(file)

if __name__ == '__main__':
    print(config['test_reuse_full'])
    full_agg_results = {}
    for path in glob.glob(config['test_reuse_full'] + "/*.csv"):
        csv_df = read_csv(path, encoding='latin-1')
        csv_df = csv_df[csv_df['MRR'].notna()]
        file_name = os.path.basename(path).split('.')[0]
        full_agg_results[file_name] = csv_df

    corr_list = []
    for k, v in full_agg_results.items():
        corr_m, p_value_m = pearsonr(list(v['F1 score']), list(v['MRR']))
        corr_t, p_value_t = pearsonr(list(v['F1 score']), list(v['Top1']))
        row = {'Name': k, 'Corr MRR': corr_m, 'p_value_m': float(p_value_m), 'Corr Top1': corr_t,
               'p_value_t': float(p_value_t)}
        corr_list.append(row)

    corr_df = pd.DataFrame(corr_list)
    corr_df.round(5).to_csv('corr.csv', index=False, float_format='%.5f')
