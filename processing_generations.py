import os
import pandas as pd

from models.roberta import Roberta
from utils.constants import ROBERTA, CONCEPTS


def drop_unsuccessful(set_df, name, verbose=True):
    failed_example = set_df['failed']
    set_df = set_df[~failed_example]
    nan_examples = (set_df['generation'].isna()) | (set_df['generation'] == '')
    set_df = set_df[~nan_examples]
    if verbose:
        print(f'{failed_example.sum()} failed examples are dropped from {name} set.')
        print(f'{nan_examples.sum()} nan examples are dropped from {name} set.')
    return set_df


def drop_no_change(set_df, name, verbose=True):
    same_example = set_df['generation'] == set_df['text']
    set_df = set_df[~same_example]
    if verbose:
        print(f'{same_example.sum()} unchanged are dropped from {name} set.')
    return set_df


def drop_duplicates(set_df, name, verbose=True):
    len_before = len(set_df)
    # s_filtered = s['generation'].apply(lambda x: str(x).lower())
    # s_filtered = set_df.drop_duplicates(keep='first', subset=['generation'])
    set_df = set_df.drop_duplicates(keep='first', inplace=False,
                                    subset=['generation', 'base_direction', 'target_direction'])
    set_df = set_df.loc[set_df.index]
    len_after = len(set_df)
    if verbose:
        print(f'{len_before - len_after} duplicates are dropped from {name} set')
    return set_df


def drop_faults_words(set_df, name, verbose=True):
    faults_words = ['review', 'reviewer', 'instructor', 'no information', 'not mentioned', 'mention', 'mentioned',
                    'there is no', 'changed', 'change']
    s_filtered = set_df['generation'].apply(lambda x: str(x).lower())
    for w in faults_words:
        s_filtered = s_filtered[~s_filtered.str.contains(w)]
    if verbose:
        print(f'{len(set_df) - len(s_filtered)} generations with faults words are dropped from {name} set')
    set_df = set_df.loc[s_filtered.index]
    return set_df


def drop_faults_ending(set_df, name, verbose=True):
    faults_ending = ['the', 'a', ',', 'however', 'unfortunately', 'however,', 'unfortunately,', 'but', 'but,']
    s_filtered = set_df['generation'].apply(lambda x: str(x).lower())
    for w in faults_ending:
        s_filtered = s_filtered[~s_filtered.str.endswith(w)]
    if verbose:
        print(f'{len(set_df) - len(s_filtered)} generations with faults words are dropped from {name} set')
    set_df = set_df.loc[s_filtered.index]
    return set_df


def dry_preprocess(loader):
    for name in loader:
        set_df = loader[name]
        lentgh_before = len(set_df)
        set_df = drop_unsuccessful(set_df, name, verbose=False)
        set_df = drop_no_change(set_df, name, verbose=False)
        set_df = drop_duplicates(set_df, name, verbose=False)
        set_df = drop_faults_words(set_df, name, verbose=False)
        set_df = drop_faults_ending(set_df, name, verbose=False)
        loader[name] = set_df
        print(f'{100 * (lentgh_before - len(set_df)) // lentgh_before}% examples are dropped from {name} set.')
    return loader


def get_aspects_predictions(loader):
    for key in loader.keys():
        s = loader[key]

        # if 'text' in s.columns:
        #     s = s.dropna(subset=['text'])
        #     text_col = 'text'
        # else:
        #     text_col = 'description'
        text_col = 'generation'
        models_paths = {concept: f'saved_models/{concept}/{ROBERTA}' for concept in CONCEPTS}
        model_to_predict_aspect = {aspect: Roberta(models_paths[aspect], num_labels=3) for aspect in CONCEPTS}
        for concept in CONCEPTS:
            s[f'{concept}_predictions_generation'] = model_to_predict_aspect[concept].get_predictions(
                list(s[text_col]), return_predictions=True, batch_size=1024)

        loader[key] = s

    return loader


def filter_by_treatment(loader):
    encode_int = {0: 'unknown', 1: 'Negative', 2: 'Positive'}
    encode_str = {'tensor(0)': 'unknown', 'tensor(1)': 'Negative', 'tensor(2)': 'Positive'}
    for name in loader:
        s = loader[name]
        for concept in CONCEPTS:
            if type(s[f'{concept}_predictions_generation'].iloc[0]) == str:
                s[f'{concept}_predictions_generation'] = s[f'{concept}_predictions_generation'].apply(
                    lambda x: encode_str[x])
            else:
                s[f'{concept}_predictions_generation'] = s[f'{concept}_predictions_generation'].apply(
                    lambda x: encode_int[int(x)])

    # keep the examples with target direction in intervention aspect = {intervention_aspect}_predictions_generation
    for name in loader:
        s = loader[name]
        idxes = []
        for idx in s.index:
            example = s.loc[idx]
            intervention_aspect = example['intervention_aspect']
            target_direction = example['target_direction']
            if target_direction == example[f'{intervention_aspect}_predictions_generation']:
                idxes.append(idx)

        print(f'{len(s) - len(idxes)} examples are dropped from {name} set, since a meaningless changing.')
        s = s.loc[idxes]
        loader[name] = s
    return loader


# def filter_by_confounder(source_loader, generation_loader):
#     for name in generation_loader:
#         source_set = source_loader[name]
#         generation_set = generation_loader[name]
#         idxes = []
#         for idx in generation_set.index:
#             example = generation_set.loc[idx]
#             intervention_aspect = example['intervention_aspect']
#             confounder = example['confounder']
#             if confounder == source_set.loc[idx, f'{intervention_aspect}_predictions']:
#                 idxes.append(idx)


def save_filtered(loader, path_dir):
    for name in loader:
        set_df = loader[name]
        set_df.to_csv(f'{path_dir}/{name}_generations.csv', index=False)


path_edited_sets = 'sets/filtered_generations'


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


# make_dir(path_edited_sets)
names = ['train', 'validation', 'test']
# generations_loader = {n: pd.read_csv(f'sets/generations/{n}_generations.csv') for n in names}
#
# filtered_loader = dry_preprocess(generations_loader)
#
# predicted_loader = get_aspects_predictions(filtered_loader)
# save_filtered(predicted_loader, path_edited_sets)
# wet_filtered_loader = wet_filter(predicted_loader)
# save_filtered(wet_filtered_loader, path_edited_sets)

loader = {n: pd.read_csv(f'{path_edited_sets}/{n}_generations.csv') for n in names}

# create full set train


# create expanded set matching
