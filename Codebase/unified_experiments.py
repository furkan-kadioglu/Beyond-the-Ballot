import json
from collections import defaultdict

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from scipy.stats import spearmanr


def drop_missing_values(df):
    return df[~df.isin([77, 88, 99]).any(axis=1)]


def process_row(row, vectors):
    country_code = row.name[:2]
    row_vector = 0

    for col, value in row.items():
        if value==77:
            row_vector += vectors[col][country_code][1]
        elif value==88:
            row_vector += vectors[col][country_code][2]
        elif value==99:
            row_vector += vectors[col][country_code][3]
        else:
            row_vector += value * vectors[col][country_code][0]

    return row_vector


def get_intervals(array, num_of_choices, cutoff=40):
    lower_bound = np.percentile(array, cutoff/2)
    upper_bound = np.percentile(array, 100-(cutoff/2))
    step_size = (upper_bound - lower_bound) / num_of_choices
    intervals = [lower_bound + i * step_size for i in range(num_of_choices + 1)]
    return intervals


def find_interval(value, intervals, choices):
    if value <= intervals[0]:
        return choices[0]
    
    if value >= intervals[-1]:
        return choices[-1]
    
    for i in range(len(intervals) - 1):
        if intervals[i] <= value < intervals[i + 1]:
            return choices[i]
        

def run_experiment_for_target_variable(target_variables, drop_missing_val=False):
    # Individual Belief Embedding Estimation 
    with open('vectors.pkl', 'rb') as f:
        vectors = pickle.load(f)

    whole_df = pd.read_csv('preprocessed_data.csv')
    whole_df.set_index('id', inplace=True)

    non_normalized_df = pd.read_csv('non_normalized_data.csv')
    non_normalized_df.set_index('id', inplace=True)

    if drop_missing_val:
        whole_df = drop_missing_values(whole_df)
        non_normalized_df = drop_missing_values(non_normalized_df)
        
    df = whole_df[whole_df.columns.difference(target_variables)]

    participants = df.apply(lambda x: process_row(x, vectors), axis=1).to_dict()
    participants = {
        p: {'embedding': v, 'projections':{}, 'predictions':{}, 'gold':{}} for p, v in participants.items()
    }

    # Projection
    distributions = defaultdict(list)

    for target in target_variables:
        for participant, values in participants.items():
            country = participant[:2]
            target_vector = vectors[target][country][0]

            length_of_projection = (
                np.dot(target_vector, values["embedding"]) / 
                np.linalg.norm(values["embedding"])
            )

            distributions[target].append(length_of_projection)

            participants[participant]["projections"][target] = length_of_projection
            participants[participant]['gold'][target] = non_normalized_df.loc[participant][target]

    # Prediction
    with open("codebook.json", "r") as codebook_file:
        codebook = json.load(codebook_file)

    distributions = {
        t: np.array([
            participants[p]["projections"][t]
            for p in participants.keys()
        ]) 
        for t in target_variables
    }
    for t in target_variables:
        choices = codebook[t]['values']
        intervals = get_intervals(distributions[t], len(choices), 0)

        for p in participants.keys():
            participants[p]['predictions'][t] = find_interval(participants[p]['projections'][t], intervals, choices)
    all_predictions = {}
    for t in target_variables:
        predictions = []
        golds = []
        for p in participants:
            if participants[p]['gold'][t] not in [77, 88, 99]:
                predictions.append(
                    participants[p]['predictions'][t]
                )
                golds.append(
                    int(participants[p]['gold'][t])
                )
        all_predictions[t] = (golds, predictions)

    results_df = pd.DataFrame(columns=['mean_absolute_error', 'cohen_kappa_score', 'spearmanr'])

    for t, (y_true, y_pred) in all_predictions.items():
        results_df.loc[t] = [
            mean_absolute_error(y_true, y_pred),
            cohen_kappa_score(y_true, y_pred),
            spearmanr(y_true, y_pred)[0]  
        ]

    return results_df
    