import pandas as pd
from datasets import load_dataset
import numpy as np
from utils.constants import CONCEPTS, DIRECTIONS
from tqdm import tqdm
import random
from utils.constants import EDITS_PATH, GENERATIONS_PATH, FILTERED_GENERATIONS_SETS, SOURCE_PATH


def load_edits_sets(seed=42, create=False):
    names = [f'train_set_{seed}', f'matching_set_{seed}']
    if not create:
        names = names + [f'expanded_matching_set_{seed}', f'full_train_set_{seed}', 'full_validation_set']
    sets = {}
    for s in names:
        df = pd.read_csv(f'{EDITS_PATH}{s}.csv')
        sets[s] = df
    return sets


def load_source_sets():
    names = ['train_exclusive', 'validation', 'test']
    sets = {}
    for s in names:
        df = pd.read_csv(f'{SOURCE_PATH}{s}.csv')
        sets[s] = df

    return sets


def load_generations_sets(filtered=True, create=False):
    names = ['train_generations', 'validation_generations', 'test_generations']
    if not create:
        names = names + ['all_generations']
    sets = {}
    for s in names:
        if filtered:
            df = pd.read_csv(f'{FILTERED_GENERATIONS_SETS}{s}.csv')
        else:
            df = pd.read_csv(f'{GENERATIONS_PATH}{s}.csv')
        df = df[(df['failed'] == False) & (~df['generation'].isna())]
        sets[s] = df

    return sets


def preprocess_hf_dataset(dataset: object, one_example_per_world: object = False, verbose: object = 0,
                          dataset_type: object = '2-way') -> object:
    """
    Preprocess the CEBaB dataset loaded from HuggingFace.
    Drop 'no majority' data, encode all labels as ints.
    """
    assert dataset_type in ['2-way', '3-way', '5-way']

    # only use one example per exogenous world setting if required
    if one_example_per_world:
        dataset['train'] = dataset['train_exclusive']
    else:
        dataset['train'] = dataset['train_inclusive']

    train = dataset['train'].to_pandas()
    dev = dataset['validation'].to_pandas()
    test = dataset['test'].to_pandas()
    train_inclusive = dataset['train_inclusive'].to_pandas()

    train_inclusive_no_majority = train_inclusive['review_majority'] == 'no majority'
    # drop no majority reviews
    train_no_majority = train['review_majority'] == 'no majority'
    if verbose:
        percentage = 100 * sum(train_no_majority) / len(train)
        print(f'Dropping no majority reviews: {round(percentage, 4)}% of train dataset.')
    train = train[~train_no_majority]
    train_inclusive = train_inclusive[~train_inclusive_no_majority]

    # encode datasets
    train = encode_dataset(train, verbose=verbose, dataset_type=dataset_type)
    train_inclusive = encode_dataset(train_inclusive, verbose=verbose, dataset_type=dataset_type)
    dev = encode_dataset(dev, verbose=verbose, dataset_type=dataset_type)
    test = encode_dataset(test, verbose=verbose, dataset_type=dataset_type)

    # fill NAs with the empty string
    aspect_columns = list(filter(lambda col: 'aspect' in col, list(train.columns)))
    train_inclusive[aspect_columns] = train_inclusive[aspect_columns].fillna('')
    train[aspect_columns] = train[aspect_columns].fillna('')
    dev[aspect_columns] = dev[aspect_columns].fillna('')
    test[aspect_columns] = test[aspect_columns].fillna('')

    return {'train_exclusive': train, 'validation': dev, 'test': test, 'train_inclusive': train_inclusive}


def encode_dataset(dataset, verbose=0, dataset_type='5-way'):
    """
    Encode the review and aspect columns.
    For 2-way experiments, drop neutral reviews.
    """
    # drop neutral in 2-way setting:
    if dataset_type == '2-way':
        neutral = dataset['review_majority'] == '3'
        dataset = dataset[~neutral]
        if verbose:
            print(f'Dropped {sum(neutral)} examples with a neutral label.')

    # encode dataset with the dataset_type
    encoding = None
    if dataset_type == '2-way':
        encoding = {
            "1": 0,
            "2": 0,
            "4": 1,
            "5": 1,
        }
    elif dataset_type == '3-way':
        encoding = {
            "1": 0,
            "2": 0,
            "3": 1,
            "4": 2,
            "5": 2
        }
    elif dataset_type == "5-way":
        encoding = {
            "1": 0,
            "2": 1,
            "3": 2,
            "4": 3,
            "5": 4
        }
    dataset['review_majority'] = dataset['review_majority'].apply(lambda score: encoding[score])
    return dataset


def load_cebab(dataset_type=None):
    ds_cebab = load_dataset('CEBaB/CEBaB')
    df_loader = {}
    if dataset_type is None:
        for name in ds_cebab:
            if name == 'train_observational':
                continue
            df_cebab = ds_cebab[name].to_pandas()
            df_loader[name] = df_cebab
    else:
        df_loader = preprocess_hf_dataset(ds_cebab, one_example_per_world=True, dataset_type=dataset_type)

    return df_loader


def get_intervention_pairs(df, dataset_type="2-way", verbose=0):
    """
    Given a dataframe in the CEBaB data scheme, return all intervention pairs.
    """
    assert dataset_type in ['2-way', '3-way', '5-way']

    # Drop label distribution and worker information.
    columns_to_keep = ['id', 'original_id', 'edit_id', 'is_original', 'edit_goal', 'edit_type', 'description',
                       'review_majority',
                       'food_aspect_majority', 'ambiance_aspect_majority', 'service_aspect_majority',
                       'noise_aspect_majority', 'opentable_metadata']
    columns_to_keep += [col for col in df.columns if 'prediction' in col]
    df = df[columns_to_keep]

    # get all the intervention pairs
    unique_originals = df.original_id.unique()
    to_merge = []
    for unique_id in unique_originals:
        df_slice = df[df['original_id'] == unique_id]
        if len(df_slice) > 1:
            pairs_slice = get_pairs_per_original(df_slice)
            to_merge.append(pairs_slice)
    pairs = pd.concat(to_merge)

    # drop unsuccessful edits
    pairs = drop_unsuccessful_edits(pairs, verbose=verbose)

    # onehot encode
    # pairs = _pairs_to_onehot(pairs, dataset_type=dataset_type)

    return pairs


def get_pairs_per_original(df):
    """
    For a df containing all examples related to one original,
    create and return all the possible intervention pairs.
    """
    assert len(df.original_id.unique()) == 1

    df_edit = df[df['is_original'] == False].reset_index(drop=True)
    if len(df_edit):
        df_original = pd.concat([df[df['is_original'] == True]] * len(df_edit)).reset_index(drop=True)
    else:
        df_original = df[df['is_original'] == True].reset_index(drop=True)

    assert (len(df_original) == 0) or (len(df_edit) == 0) or (len(df_edit) == len(df_original))

    # (edit, original) pairs
    edit_original_pairs = None
    original_edit_pairs = None
    if len(df_original) and len(df_edit):
        df_edit_base = df_edit.rename(columns=lambda x: x + '_base')
        df_original_counterfactual = df_original.rename(columns=lambda x: x + '_counterfactual')

        edit_original_pairs = pd.concat([df_edit_base, df_original_counterfactual], axis=1)

        # (original, edit) pairs
        df_edit_counterfactual = df_edit.rename(columns=lambda x: x + '_counterfactual')
        df_original_edit = df_original.rename(columns=lambda x: x + '_base')

        original_edit_pairs = pd.concat([df_original_edit, df_edit_counterfactual], axis=1)

    # (edit, edit) pairs
    edit_edit_pairs = None
    if len(df_edit):
        # The edits are joined based on their edit type.
        # Actually, the 'edit_type' can also differ from the edit performed, but there is no clean way of resolving this.
        edit_edit_pairs = df_edit.merge(df_edit, on='edit_type', how='inner', suffixes=('_base', '_counterfactual'))
        edit_edit_pairs = edit_edit_pairs[edit_edit_pairs['id_base'] != edit_edit_pairs['id_counterfactual']]
        edit_edit_pairs = edit_edit_pairs.rename(columns={'edit_type': 'edit_type_base'})
        edit_edit_pairs['edit_type_counterfactual'] = edit_edit_pairs['edit_type_base']
    if edit_original_pairs is not None:
        edit_original_pairs.reset_index(drop=True)
        # get all pairs
        pairs = pd.concat([edit_original_pairs, original_edit_pairs, edit_edit_pairs]).reset_index(drop=True)
    else:
        pairs = pd.concat([original_edit_pairs, edit_edit_pairs]).reset_index(drop=True)
    # annotate pairs with the intervention type and the direction (calculated from the validated labels)
    pairs = _get_intervention_type_and_direction(pairs)

    return pairs


def drop_unsuccessful_edits(pairs, verbose=True):
    """
    Drop edits that produce no measured aspect change.
    """
    # Make sure the validated labels of the edited aspects are different.
    # We can not do this comparison based on 'edit_goal_*' because the final label might differ from the goal.
    meaningless_edits = pairs['intervention_aspect_base'] == pairs['intervention_aspect_counterfactual']
    if verbose:
        print(
            f'Dropped {sum(meaningless_edits)} pairs that produced no validated label change.'
            f' This is due to faulty edits by the workers or edits with the same edit_goal.')
    pairs = pairs[~meaningless_edits]

    return pairs


def _get_intervention_type_and_direction(pairs):
    """
    Annotate a dataframe of pairs with their invention type
    and the validated label of that type for base and counterfactual.
    """
    # get intervention type
    pairs.loc[:, 'intervention_type'] = np.maximum(pairs['edit_type_base'].astype(str),
                                                   pairs['edit_type_counterfactual'].astype(str))

    for idx in pairs.index:
        pairs.loc[idx, 'intervention_aspect_base'] = pairs.loc[idx,
                                                               f'{pairs.loc[idx, "intervention_type"]}_aspect_majority_base']
        pairs.loc[idx, 'intervention_aspect_counterfactual'] = pairs.loc[idx,
                                                                         f'{pairs.loc[idx, "intervention_type"]}_aspect_majority_counterfactual']

    # get base/counterfactual value of the intervention aspect
    # pairs['intervention_aspect_base'] = pairs[f'{}']
    #     ((pairs['intervention_type'] == 'ambiance') * pairs['ambiance_aspect_majority_base']) + \
    #     ((pairs['intervention_type'] == 'noise') * pairs['noise_aspect_majority_base']) + \
    #     ((pairs['intervention_type'] == 'service') * pairs['service_aspect_majority_base']) + \
    #     ((pairs['intervention_type'] == 'food') * pairs['food_aspect_majority_base'])
    #
    # pairs['intervention_aspect_counterfactual'] = \
    #     ((pairs['intervention_type'] == 'ambiance') * pairs['ambiance_aspect_majority_counterfactual']) + \
    #     ((pairs['intervention_type'] == 'noise') * pairs['noise_aspect_majority_counterfactual']) + \
    #     ((pairs['intervention_type'] == 'service') * pairs['service_aspect_majority_counterfactual']) + \
    #     ((pairs['intervention_type'] == 'food') * pairs['food_aspect_majority_counterfactual'])

    return pairs


def _pairs_to_onehot(pairs, dataset_type="5-way"):
    """
    Cast the review majority columns to onehot vectors.
    """
    rng = None
    if dataset_type == '2-way':
        rng = range(0, 2)
    elif dataset_type == '3-way':
        rng = range(0, 3)
    elif dataset_type == '5-way':
        rng = range(0, 5)
    pairs['review_majority_counterfactual'] = _int_to_onehot(pairs['review_majority_counterfactual'], rng)
    pairs['review_majority_base'] = _int_to_onehot(pairs['review_majority_base'], rng)

    return pairs


def _int_to_onehot(series, rng):
    """
    Encode a series of ints as a series of onehot vectors.
    Assumes the series of ints is contained within the range.
    """
    offset = rng[0]
    rng = max(rng) - min(rng) + 1

    def _get_onehot(x):
        zeros = np.zeros(rng)
        zeros[int(x) - offset] = 1.0
        return zeros

    return series.apply(_get_onehot)


def batchify_eval(df_generations, df_source, treatment, tokenizer, approach='all'):
    # I want my batch to be in the following format:
    # text, treatment counterfactuals, confounder counterfactuals, negative samples, approx samples
    # add columns of tokenization to the df_source
    tokenized_base = tokenizer(list(df_source['description']), return_tensors='pt', padding=True, truncation=True)
    for k in tokenized_base.keys():
        df_source.loc[:, k] = list(tokenized_base[k])

    # df_source.loc[:, tokenized_base.keys()] =
    source_columns_to_keep = list(tokenized_base.keys()) + ['description', 'original_id'] + [
        f'{concept}_aspect_majority' for concept in CONCEPTS]
    df_source = df_source[source_columns_to_keep]
    # df_generations.loc[:, 'tokenized_generation'] = tokenizer(list(df_generations['generation']), return_tensors='pt', padding=True, truncation=True)
    tokenized_generation = tokenizer(list(df_generations['generation']), return_tensors='pt', padding=True,
                                     truncation=True)
    for k in tokenized_generation.keys():
        df_generations.loc[:, k] = list(tokenized_generation[k])

    confounders = [concept for concept in CONCEPTS if concept != treatment]
    batches = []
    for idx in df_source.index:
        batch = {}
        example = df_source.loc[idx]
        generations_slice = df_generations[(df_generations['original_id'] == example['original_id'])]
        generations_slice = generations_slice[df_generations['text'] == example['description']]
        treatment_counterfactuals = generations_slice[generations_slice['intervention_aspect'] == treatment]
        if len(treatment_counterfactuals) == 0:
            continue
        for confounder in confounders:
            treatment_counterfactuals = treatment_counterfactuals[
                treatment_counterfactuals[f'{confounder}_predictions_generation'] == example[
                    f'{confounder}_aspect_majority']]
        if len(treatment_counterfactuals) == 0:
            continue
        # the complement of the treatment counterfactuals are the confounder counterfactuals
        confounder_counterfactuals = generations_slice.drop(treatment_counterfactuals.index)
        confounder_counterfactuals = confounder_counterfactuals[confounder_counterfactuals['intervention_aspect'].isin(
            confounders)]

        if len(confounder_counterfactuals) == 0:
            continue
        # confounder_counterfactuals = confounder_counterfactuals[confounder_counterfactuals[f'{treatment}_label'] == example[
        #     f'{treatment}_aspect_majority']]

        candidates = df_source.drop(idx)
        candidates = candidates[candidates['original_id'] != example['original_id']]
        # approx examples are examples that sharing the same confounding value as the given example
        temp = candidates.copy()
        for confounder in confounders:
            temp = temp[temp[f'{confounder}_aspect_majority'] == example[f'{confounder}_aspect_majority']]
        approx_examples = temp
        # negative examples are the complement of approx examples
        negative_examples = candidates.drop(approx_examples.index)
        batch['query'] = example
        batch['t_cf'] = treatment_counterfactuals
        batch['c_cf'] = confounder_counterfactuals
        batch['negative'] = negative_examples
        batch['approx'] = approx_examples
        if approach == 'all':
            if (len(treatment_counterfactuals) == 0) or (len(confounder_counterfactuals) == 0) or (
                    len(negative_examples) == 0 or len(approx_examples) == 0):
                continue

        batches.append(batch)

    return batches


def batchify_train(full_set, treatment, tokenizer, approach='all', generation_index='-1'):
    # I want my batch to be in the following format:
    # text, treatment counterfactuals, confounder counterfactuals, negative samples, approx samples
    # add columns of tokenization to the df_source
    tokenized_base = tokenizer(list(full_set['text']), return_tensors='pt', padding=True, truncation=True)
    for k in tokenized_base.keys():
        full_set.loc[:, k] = list(tokenized_base[k])

    # df_source.loc[:, tokenized_base.keys()] =

    confounders = [concept for concept in CONCEPTS if concept != treatment]
    batches = []
    generation_index_lst = []
    options = ['-1', '0', '1', '2', '3']
    for option in options:
        if option in generation_index:
            generation_index_lst.append(int(option))
            generation_index = generation_index.replace(option, '')

    queries_indexes = full_set[full_set['generation_index'].isin(generation_index_lst)].index
    for idx in queries_indexes:

        generation_idx = full_set.loc[idx]['generation_index']
        query = full_set.loc[idx]
        if len(query['text']) <= 2:
            continue
        cfs_slice = full_set[full_set['original_id'] == query['original_id']].drop(idx)

        # drop all the cfs with the same text as the query
        cfs_slice = cfs_slice[cfs_slice['text'] != query['text']]
        # sources_cfs
        sources_cfs = cfs_slice[cfs_slice['generation_index'] == -1]

        # confounder counterfactuals candidates
        treatment_equal = cfs_slice[cfs_slice[f'{treatment}_predictions'] == query[f'{treatment}_predictions']]
        treatment_equal = treatment_equal[
            treatment_equal['intervention_aspect'].isin(confounders)]
        confounder_cf_candidates = pd.concat([treatment_equal, sources_cfs[
            sources_cfs[f'{treatment}_predictions'] == query[f'{treatment}_predictions']]])

        # treatment counterfactuals candidates
        treatment_different = cfs_slice[cfs_slice[f'{treatment}_predictions'] != query[f'{treatment}_predictions']]
        treatment_different = treatment_different[treatment_different['intervention_aspect'] == treatment]
        treatment_cf_candidates = pd.concat([treatment_different, sources_cfs[
            sources_cfs[f'{treatment}_predictions'] != query[f'{treatment}_predictions']]])

        # making sure all the confounders are the same in treatment-different set
        for confounder in confounders:
            treatment_cf_candidates = treatment_cf_candidates[
                treatment_cf_candidates[f'{confounder}_predictions'] == query[f'{confounder}_predictions']]

        rows_to_drop = []
        for idx_c_cf in confounder_cf_candidates.index:
            count = 0
            for confounder in confounders:
                if confounder_cf_candidates.loc[idx_c_cf, f'{confounder}_predictions'] == query[
                    f'{confounder}_predictions']:
                    count += 1
            if count != (len(confounders) - 1):
                rows_to_drop.append(idx_c_cf)
        confounder_cf_candidates = confounder_cf_candidates.drop(rows_to_drop)

        candidates = full_set.drop(idx)
        temp = candidates.copy()
        temp = temp[(temp['original_id'] != query['original_id'])]
        # approx examples are examples that sharing the same confounding value as the given example
        for confounder in confounders:
            temp = temp[temp[f'{confounder}_predictions'] == query[f'{confounder}_predictions']]
        approx_examples = temp
        # negative examples are the complement of approx examples
        negative_examples = candidates.drop(approx_examples.index)
        batch = {}

        batch['query'] = query
        batch['c_cf'] = confounder_cf_candidates
        batch['negative'] = negative_examples
        batch['approx'] = approx_examples
        batch['t_cf'] = treatment_cf_candidates
        # for t_cf_idx in treatment_counterfactuals.index:
        #     batch['t_cf'] = pd.DataFrame(treatment_counterfactuals.loc[t_cf_idx]).T
        if approach == 'all':
            if (len(treatment_cf_candidates) == 0) or (len(confounder_cf_candidates) == 0) or (
                    len(negative_examples) == 0 or len(approx_examples) == 0):
                continue
        assert len(negative_examples) + len(approx_examples) == len(candidates)
        batches.append(batch)

    random.shuffle(batches)
    return batches


def create_all_sets_generations(train_generations, valid_generations, test_generations,
                                path='sets/generations/all_sets_generations.csv'):
    # add a column for split type and concatenate all the sets
    train_generations['split'] = 'train'
    valid_generations['split'] = 'valid'
    test_generations['split'] = 'test'
    all_sets = pd.concat([train_generations, valid_generations, test_generations])
    all_sets = all_sets.reset_index(drop=True)
    all_sets.to_csv(path)
    return all_sets
