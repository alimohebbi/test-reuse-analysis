from math import floor

import pandas
import pandas as pd
import yaml
from pandas import read_csv

from util import make_df_plot_friendly

with open(r'../config.yaml') as file:
    config = yaml.safe_load(file)


def load_results() -> pandas.DataFrame:
    result = read_csv(config['semantic_matching_results'] + 'latest_all.csv')
    result = make_df_plot_friendly(result)
    return result


def get_frequencies(results, metric):
    results.sort_values(by=[metric], ascending=False, inplace=True)
    df_size = results.shape[0]
    columns = ['algorithm', 'descriptors', 'training_set', 'word_embedding']
    percentiles = [1, 5, 10]
    d = {'instances': [], 1: [], 5: [], 10: []}
    for column in columns:
        values = results[column].unique()
        d['instances'].extend(values)
        for percentile in percentiles:
            for value in values:
                top_n = floor(percentile / 100 * df_size)
                top_n_rows = results.head(top_n)
                frequency = top_n_rows.query(f'{column} == "{value}"').shape[0] / top_n * 100
                d[percentile].append(frequency)
    return pd.DataFrame(d)


if __name__ == '__main__':
    results = load_results()
    df = get_frequencies(results, 'MRR')
    df.to_csv('frequency_mrr.csv', index=False)
    print(df)
    df = get_frequencies(results, 'top1')
    df.to_csv('frequency_top1.csv', index=False)
    print(df)
