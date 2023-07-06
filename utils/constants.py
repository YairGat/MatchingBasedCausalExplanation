import os
from datetime import datetime

import torch


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


DATE = datetime.now().strftime("%d_%m_%Y_%H_%M")
RESULTS_PATH = 'outputs'
make_dir(RESULTS_PATH)
GENERATIONS_PATH = 'sets/generations/'
SOURCE_PATH = 'sets/sources/'
FILTERED_GENERATIONS_SETS = 'sets/filtered_generations/'
EDITS_PATH = 'sets/edits/'
SAVED_TRAINED_MODELS_PATH = 'saved_models'
BERT_SENTIMENT_PATH = os.path.join(SAVED_TRAINED_MODELS_PATH, 'overall_sentiment_bert', 'bert-base-uncased')
DISTIL_BERT_SENTIMENT_PATH = os.path.join(SAVED_TRAINED_MODELS_PATH, 'overall_sentiment_distil_bert',
                                          'distilbert-base-uncased')

ROBERTA_SENTIMENT_PATH = os.path.join(SAVED_TRAINED_MODELS_PATH, 'overall_sentiment_roberta',
                                      'roberta-base')

CONCEPTS = ['food', 'ambiance', 'service', 'noise']
POSITIVE = 'Positive'
NEGATIVE = 'Negative'
UNKNOWN = 'unknown'
DIRECTIONS = [NEGATIVE, UNKNOWN, POSITIVE]
ROBERTA = 'roberta-base'
BERT = 'bert-base-uncased'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
