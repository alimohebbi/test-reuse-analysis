import os
from os import listdir

import pandas as pd


def rename_subjects(x):
    res = x['mig_name'].replace('ExpenseTracker1', 'a61-b61')
    res = res.replace('ExpenseTracker2', 'a62-b61')
    res = res.replace('ExpenseTracker3', 'a63-b61')
    res = res.replace('ExpenseTracker4', 'a64-b61')

    res = res.replace('NoteTaking2', 'a72-b71')
    res = res.replace('NoteTaking3', 'a73-b71')
    res = res.replace('NoteTaking4', 'a74-b71')

    res = res.replace('ShoppingList1', 'a81-b81')
    res = res.replace('ShoppingList2', 'a82-b81')
    res = res.replace('ShoppingList3', 'a83-b81')
    res = res.replace('ShoppingList4', 'a84-b81')
    return res


def convert_config_names(df):
    if 'config' not in df.columns:
        df.reset_index(inplace=True)
    df = df.rename(
        columns={'index': 'config', 'tp': 'TP', 'fp': 'FP', 'tn': 'TN', 'fn': 'FN', 'effort_leveneshtein': 'E.L.',
                 'effort_damerau_levenshtein': 'E.D.L', 'accuracy': 'Accuracy', 'precision': 'Precision',
                 'recall': 'Recall', 'f1_score': 'F1 score', 'top1': 'Top1'})

    df['config'] = df['config'].apply(lambda x: x.replace('edit_distance', 'es'))
    df['config'] = df['config'].apply(lambda x: x.replace('jaccard', 'js'))
    df['Training'] = df['config'].apply(lambda x: x.split('_')[1])
    df['Embedding'] = df['config'].apply(lambda x: x.split('_')[0])
    df['Descriptors'] = df['config'].apply(lambda x: x.split('_')[3])
    df['Algorithm'] = df['config'].apply(lambda x: x.split('_')[2])
    df['Descriptors'] = df['Descriptors'].apply(lambda x: x.replace('craftdroid', 'craftdroid_d'))
    df['Descriptors'] = df['Descriptors'].apply(lambda x: x.replace('atm', 'atm_d'))
    df['Algorithm'] = df['Algorithm'].apply(lambda x: x.replace('atm', 'atm_a'))
    df['Algorithm'] = df['Algorithm'].apply(lambda x: x.replace('craftdroid', 'craftdroid_a'))
    df['Algorithm'] = df['Algorithm'].apply(lambda x: x.replace('custom', 'semfinder_a'))
    df['Algorithm'] = df['Algorithm'].apply(lambda x: x.replace('adaptdroid', 'adaptdroid_a'))
    df['Training'] = df['Training'].apply(lambda x: x.replace('android', 'manuals'))
    return df


def reorder_columns(df):
    df.drop('config', inplace=True, axis=1)
    first_column = df.pop('Algorithm')
    df.insert(0, 'Algorithm', first_column)
    second = df.pop('Descriptors')
    df.insert(1, 'Descriptors', second)
    third = df.pop('Embedding')
    df.insert(2, 'Embedding', third)
    forth = df.pop('Training')
    df.insert(3, 'Training', forth)
    last = df.pop('MRR')
    df.insert(16, 'MRR', last)
    forth = df.pop('reduction_leveneshtein')
    df.insert(10, 'R.L.', forth)
    last = df.pop('reduction_damerau_leveneshtein')
    df.insert(11, 'R.D.L', last)
    df = df.rename(columns={'index': 'config'})
    return df


def make_config_column(df):
    data = df.copy()
    data = convert_config_names(data)
    data['config'] = data['Algorithm'] + '_' + data['Descriptors'] + '_' + data['Embedding'] + '_' + data[
        'Training']
    data.drop(columns=['Algorithm', 'Descriptors', 'Embedding', 'Training'], inplace=True)
    return data


def add_mig_name(df):
    df = df.copy()
    if 'task' not in df.columns:
        df['task'] = ''
    df['mig_name'] = ''
    df['mig_name'] = df.apply(lambda x: mig_name_getter(x), axis =1)
    return df


def mig_name_getter(x):
    return x['src_app'] + ' - ' + x['target_app'] + ' - ' + x['task']


def add_unified_mig_name(df):
    df = add_mig_name(df)
    df['mig_name'] = df.apply(rename_subjects, axis=1)
    return df


def add_file_name_as_config(csv, path):
    file_name = os.path.basename(path).split('.')[0]
    file_name = file_name.split('result_')[1]
    if 'random' in file_name:
        file_name = 'NA_NA_random_NA'
    if 'perfect' in file_name:
        file_name = 'NA_NA_perfect_NA'
    csv['config'] = file_name
    return csv

def concat_config_results(path):
    results_fname = [f for f in listdir(path) if '.csv' in f]
    all_results = []
    for fname in results_fname:
        result_f = pd.read_csv(path + fname)
        add_file_name_as_config(result_f, fname)
        all_results.append(result_f)
    all_results_df = pd.concat(all_results)
    all_results_df = all_results_df[['src_app', 'target_app', 'f1_score', 'config']]
    return all_results_df.sort_values(by=['src_app', 'target_app'])


