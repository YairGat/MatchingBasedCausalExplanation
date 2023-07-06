import os

from transformers import TrainingArguments
from utils.constants import POSITIVE, NEGATIVE, UNKNOWN, SAVED_TRAINED_MODELS_PATH
from datasets import Dataset, load_dataset
import pyarrow
import pandas as pd
from datasets import load_dataset


def train_aspect_model(classifier, aspect, df_train, df_dev, df_test):
    path_to_save = os.path.join(SAVED_TRAINED_MODELS_PATH, aspect)
    # remove non-assigned values
    encoding = {NEGATIVE: 1, POSITIVE: 2, UNKNOWN: 0}

    df_train = df_train[df_train[f'{aspect}_aspect_majority'].isin([NEGATIVE, POSITIVE, UNKNOWN])]
    df_train[f'{aspect}_aspect_majority'] = df_train[f'{aspect}_aspect_majority'].apply(lambda score: encoding[score])
    train_ds = Dataset(pyarrow.Table.from_pandas(df_train))

    df_dev = df_dev[df_dev[f'{aspect}_aspect_majority'].isin([NEGATIVE, POSITIVE, UNKNOWN])]

    df_dev[f'{aspect}_aspect_majority'] = df_dev[f'{aspect}_aspect_majority'].apply(lambda score: encoding[score])
    dev_ds = Dataset(pyarrow.Table.from_pandas(df_dev))

    df_test = df_test[df_test[f'{aspect}_aspect_majority'].isin([NEGATIVE, POSITIVE, UNKNOWN])]
    df_test[f'{aspect}_aspect_majority'] = df_test[f'{aspect}_aspect_majority'].apply(lambda score: encoding[score])
    test_ds = Dataset(pyarrow.Table.from_pandas(df_test))

    tokenized_train, tokenized_valid, tokenized_test = classifier.tokenize_sets([train_ds, dev_ds, test_ds],
                                                                                label_column=f'{aspect}_aspect_majority')
    classifier.train(aspect=aspect, tokenized_train=tokenized_train, tokenized_valid=tokenized_valid,
                     tokenized_test=tokenized_test,
                     training_args=TrainingArguments(evaluation_strategy='epoch',
                                                     learning_rate=0.00005, output_dir='outputs'),
                     path_to_save=path_to_save)
    return classifier


def tokenization_for_training(tokenizer, splits, label_column):
    """
        Tokenize the set in "splits" for training. label column is the column name of the labels in split.
    """

    def tokenize_function(example):
        return tokenizer(example['description'], truncation=True, padding=True)

    tokenized_sets = []
    for s in splits:
        if len(s) == 0:
            tokenized_sets.append([])
            continue
        tokenized_dataset = s.map(tokenize_function, batched=True)
        columns_to_remove = s.column_names
        columns_to_remove.remove(label_column)
        tokenized_dataset = tokenized_dataset.rename_column(label_column, 'labels')
        tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
        tokenized_dataset = tokenized_dataset.with_format('torch')
        tokenized_sets.append(tokenized_dataset)
    return tokenized_sets


def tokenization_for_training_gpt(tokenizer, splits, label_column):
    """
        Tokenize the set in "splits" for training. label column is the column name of the labels in split.
    """

    def tokenize_function(example):
        return tokenizer(example['description'], truncation=True, padding=True)

    tokenized_sets = []
    for s in splits:
        if len(s) == 0:
            tokenized_sets.append([])
            continue
        tokenized_dataset = s.map(tokenize_function, batched=True)
        columns_to_remove = s.column_names
        columns_to_remove.remove(label_column)
        tokenized_dataset = tokenized_dataset.rename_column(label_column, 'labels')
        tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
        tokenized_dataset = tokenized_dataset.with_format('torch')
        tokenized_sets.append(tokenized_dataset)
    return tokenized_sets


def train_sentiment_classifier(clf, df_train, df_validation, df_test, num_labels, model_description,
                               aspect='overall_sentiment'):
    sets = get_data_set_cebab([df_train, df_validation, df_test],
                              dataset_type=f'{num_labels}-way', drop_no_majority=True)

    tokenized_train, tokenized_valid, tokenized_test = clf.tokenize_sets(sets)

    # model.train_classifier_new(tokenized_train, tokenized_valid)
    training_args = TrainingArguments(evaluation_strategy='epoch', num_train_epochs=6,
                                      output_dir=SAVED_TRAINED_MODELS_PATH)
    clf.train(tokenized_train=tokenized_train, tokenized_valid=tokenized_valid, tokenized_test=tokenized_test,
              training_args=training_args, aspect=aspect, path_to_save=f'{aspect}_{model_description}')
    return clf


def get_data_set_cebab(df_cebab, dataset_type, drop_no_majority=False):
    splits = []
    encoding = None
    if dataset_type == '2-way':
        encoding = {
            1: 0,
            2: 0,
            4: 1,
            5: 1,
        }
    elif dataset_type == '3-way':
        encoding = {
            '1': 0,
            '2': 0,
            '3': 1,
            '4': 2,
            '5': 2
        }
    elif dataset_type == "5-way":
        encoding = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4
        }

    for s in df_cebab:
        if len(s) == 0:
            splits += [[]]
            continue
        if drop_no_majority:
            s = s[s['review_majority'] != 'no majority']
        s['review_majority'] = s['review_majority'].apply(lambda x: encoding[int(x)])
        ds = Dataset(pyarrow.Table.from_pandas(s))
        if 'text' in ds.column_names:
            ds = ds.rename_column('text', 'description')
        if 'edit_rating' in ds.column_names:
            ds = ds.rename_column('edit_rating', 'review_majority')
        splits += [ds]
    return splits
