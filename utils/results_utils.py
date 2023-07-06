import os

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import json
from utils.constants import DIRECTIONS, RESULTS_PATH, DATE


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


# def calculate_cebab_score_wandb(model_to_explain, concepts, explainers, pairs, name, wandb_on=True, to_return=False):
#     print('\n CEBaB score calculation has started now.')
#     explainers_descriptions = [e.get_explainer_description() for e in explainers]
#     # For each concept we provide an explanation.
#     total_log = {}
#     for concept in concepts:
#         progress_bar = tqdm(explainers)
#         for e in progress_bar:
#             progress_bar.set_description(f'Calculate the cebab score for explainer: {e.get_explainer_description()}')
#             sum_icace = []
#             for base_direction in DIRECTIONS:
#                 for target_direction in DIRECTIONS:
#                     if target_direction == base_direction:
#                         continue
#
#                     icace_error = e.icace_error(pairs=pairs, model_to_explain=model_to_explain, concept=concept,
#                                                 base_direction=base_direction, target_direction=target_direction,
#                                                 save_matches=False)
#                     icace_error = np.round(icace_error, decimals=2)
#                     description = f'\n\t Explainer: {e.get_explainer_description()}\n\t' \
#                                   f'Explained model: {model_to_explain.get_model_description()}\n\t ' \
#                                   f'Direction: {base_direction} -> {target_direction}\n' \
#                                   f'########### ICACE-error: {icace_error} ###########\n\n'
#                     sum_icace.append(icace_error)
#
#             average = round(np.mean(sum_icace), 2)
#             log = {f'{name}_{e.get_explainer_description()}': round(average, 2)}
#
#             if wandb_on:
#                 wandb.log(log)
#             total_log = {**total_log, **log}
#             print(total_log)
#     if to_return:
#         return total_log


def calculate_cebab_score(model_to_explain, concepts, explainers, pairs,
                          name, wandb_on, path_dir=None, return_log=False, save_outputs=False):
    print('\nThe pipeline has started now.')
    explainers_descriptions = [e.get_explainer_description() for e in explainers]
    df_overall = pd.DataFrame(columns=concepts, index=explainers_descriptions)

    pairs[f'prediction_base'] = model_to_explain.get_predictions(list(pairs['description_base'].values))
    pairs[f'prediction_counterfactual'] = model_to_explain.get_predictions(
        list(pairs['description_counterfactual'].values))

    for concept in tqdm(concepts):
        print(f'\nTreatment: {concept}')
        directions = [f'{b}->{t}' for b in DIRECTIONS for t in DIRECTIONS if b != t] + ['average']
        df = pd.DataFrame(columns=directions, index=explainers_descriptions)

        valid_pairs = {e_d: 0 for e_d in explainers_descriptions}
        optional_pairs = {e_d: 0 for e_d in explainers_descriptions}
        for base_direction in DIRECTIONS:
            for target_direction in DIRECTIONS:
                if target_direction == base_direction:
                    continue
                # f = open(os.path.join(results_path, f'{concept}_{base_direction}_{target_direction}.txt'), 'a')
                intervention_type_col = 'intervention_type'
                intervention_aspect_base_col = 'intervention_aspect_base'
                intervention_aspect_counterfactual_col = 'intervention_aspect_counterfactual'

                pairs_prime = pairs[
                    (pairs[intervention_type_col] == concept) & (
                            pairs[intervention_aspect_base_col] == base_direction) & (
                            pairs[intervention_aspect_counterfactual_col] == target_direction)]

                for e in explainers:
                    optional_pairs[e.get_explainer_description()] += len(pairs_prime)
                    pairs_results = e.icace_error(pairs=pairs_prime, model=model_to_explain, concept=concept,
                                                  base_direction=base_direction, target_direction=target_direction,
                                                  save_outputs=save_outputs)
                    valid_pairs[e.get_explainer_description()] += len(pairs_results)
                    icace_error = pairs_results['icace_error'].mean()
                    if icace_error is None:
                        continue
                    icace_error = np.round(icace_error, decimals=2)
                    description = f'\n\t Explainer: {e.get_explainer_description()}\n\t' \
                                  f'Explained model: {model_to_explain.get_model_description()}\n\t ' \
                                  f'Direction: {base_direction} -> {target_direction}\n' \
                                  f'########### ICACE-error: {icace_error} ###########\n\n'
                    df.loc[e.get_explainer_description(), f'{base_direction}->{target_direction}'] = round(icace_error,
                                                                                                           2)

        # f.write(description)
        df['average'] = df.mean(axis=1, skipna=True)
        df.style.highlight_max(color='lightgreen', axis=0)
        if path_dir is not None:
            results_path = get_results_path_per_concept(path_dir=path_dir, concept=concept, name=name)
            df = df.round(2)
            df.to_csv(f'{results_path}.csv')
        df_overall[concept] = df['average']

    df_overall['average'] = round(df_overall.mean(axis=1), 2)
    df_overall.style.highlight_max(color='lightgreen', axis=0)
    if path_dir is not None:
        p = get_results_path_per_concept(path_dir=path_dir, concept='overall', name=name)
        df_overall.to_csv(f'{p}.csv')

    wandb_dict = {f'{name}_{e.get_explainer_description()}': df_overall.loc[e.get_explainer_description(), 'average']
                  for e
                  in explainers}

    if wandb_on:
        wandb.log(wandb_dict)
        # wandb.log(
        #     {f'above_the_threshold_{e.get_explainer_description()}': 100 * valid_pairs[e.get_explainer_description()] /
        #                                                              optional_pairs[e.get_explainer_description()]
        #      for e in explainers})
        if return_log:
            return wandb_dict

    return df_overall


def get_results_path_per_concept(path_dir, concept, name):
    p = os.path.join(path_dir, 'cebab_score')
    make_dir(p)
    p = os.path.join(p, name)
    make_dir(p)
    p = os.path.join(p, concept)
    return p


def get_results_path(config, save_config=True, loss_version='0', ablations=False):
    results_dir = RESULTS_PATH
    results_dir = os.path.join(results_dir, config.treatment)
    make_dir(RESULTS_PATH)
    results_dir = os.path.join(results_dir, DATE)
    make_dir(results_dir)
    if loss_version == '0':
        specific_path = 'bla'
        if ablations:
            specific_path = 'ablations'
    else:
        specific_path = f't_cf-{config.t_cf}_c_cf-{config.c_cf}app-{config.approx}_negative-{config.negative}_treatment-{config.treatment}_seed-{config.seed}'
    results_dir = os.path.join(results_dir, specific_path)
    make_dir(results_dir)

    if save_config:
        with open(os.path.join(results_dir, 'config.txt'), "w") as text_file:
            json.dump(dict(config), text_file)

    return results_dir
