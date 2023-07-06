from sklearn.model_selection import train_test_split

from models.bert import Bert
from models.roberta import Roberta
from utils.results_utils import make_dir
from utils.constants import BERT, ROBERTA, CONCEPTS
import pandas as pd
from utils.data_utils import load_source_sets, load_cebab
from utils.training_utils import train_aspect_model, train_sentiment_classifier
from copy import deepcopy

RANDOM_SEED = 42
df_loader = load_cebab(dataset_type="5-way")

source_sets_path = 'sets/sources/'
make_dir(source_sets_path)

for name in df_loader.keys():
    df_loader[name].to_csv(source_sets_path + name + '.csv', index=False)

train_exclusive = df_loader['train_exclusive']

edited_sources_path = 'sets/edits/'
make_dir(edited_sources_path)

train_set, matching_set = train_test_split(train_exclusive, test_size=0.5, random_state=RANDOM_SEED)

train_set.to_csv(edited_sources_path + f'train_set_{RANDOM_SEED}.csv', index=False)
matching_set.to_csv(edited_sources_path + f'matching_set_{RANDOM_SEED}.csv', index=False)

# train aspects model if needed and load if already trained

# get predictions

# save sets
