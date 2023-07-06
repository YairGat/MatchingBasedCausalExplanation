import random

import numpy as np
import torch
import wandb
import sys
from explainers.matching import Matching
from models.bert import Bert
from models.matching_rep import MatchingRepresentation
from utils.data_utils import load_source_sets, load_generations_sets, get_intervention_pairs, batchify_eval, \
    batchify_train, load_edits_sets
from utils.model_utils import get_fine_tuned_sentiment_model
from utils.results_utils import make_dir, get_results_path


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


wandb.init()
config = wandb.config
wandb_on = True
loss_version = config.loss_version

# if (config.c_cf == 0) and (config.negative == 0):
#     if wandb_on:
#         wandb.finish()
#     sys.exit("Exiting the code with sys.exit()! division by zero")
# if (config.t_cf == 0) and (config.approx == 0):
#     if wandb_on:
#         wandb.finish()
#     sys.exit("Exiting the code with sys.exit()! no learning")

results_dir = get_results_path(config, save_config=True, ablations=True)
# results_dir = 'outputs'

set_seed(config.seed)
generations = load_generations_sets(filtered=True)
sets = load_source_sets()
edited_sets = load_edits_sets(seed=config.seed)
bert = Bert()
bert_lm = bert.get_representation_model()
tokenizer = bert.get_tokenizer()

train_batch = batchify_train(full_set=edited_sets[f'full_train_set_{config.seed}'], treatment=config.treatment,
                             tokenizer=tokenizer, generation_index=config.generation_index)

# train_batch = batchify(df_generations=generations['train_generations'], df_source=sets[f'train_set_{config.seed}'],
#                        treatment=config.treatment, tokenizer=tokenizer)
# validation_batch = batchify_eval(df_generations=generations['validation_generations'], df_source=sets['validation'],
#                                  treatment=config.treatment, tokenizer=tokenizer)

validation_batch = batchify_train(full_set=edited_sets[f'full_validation_set'], treatment=config.treatment,
                                  tokenizer=tokenizer, generation_index=config.generation_index)
cebab_eval = None

# initilaize matching rep
matching_rep = MatchingRepresentation(tokenizer=tokenizer, pretrained_model=bert_lm)
set_to_match = edited_sets[f'expanded_matching_set_{config.seed}']

matching_explainer_1 = Matching(approach=1, set_to_match=set_to_match,
                                representation_model=matching_rep, assign=False, top_k=1, threshold=0)

explainers = [matching_explainer_1]
pairs_validation = get_intervention_pairs(df=sets['validation'], dataset_type="5-way", verbose=1)
pairs_test = get_intervention_pairs(df=sets['test'], dataset_type="5-way", verbose=1)
if config.model_to_explain == 'bert':
    model_to_explain = get_fine_tuned_sentiment_model('bert')
cebab_eval = {
    'model_to_explain': model_to_explain,
    'explainers': explainers,
    'pairs_validation': pairs_validation,
    'pairs_test': pairs_test,
    'matching_description': matching_explainer_1.get_explainer_description()
}

# cebab_eval = None

# train matching rep
matching_rep.train(config, results_dir=results_dir, wandb_on=wandb_on, train_batch=train_batch,
                   valid_batch=validation_batch,
                   test_batch=None, mean_loss=config.mean_loss,
                   cebab_eval=cebab_eval, loss_version=loss_version)
