from tqdm import tqdm
from transformers import AutoTokenizer

from explainers.approx import Approx
from explainers.generative import Generative
from explainers.matching import Matching
from explainers.random_matching import RandomMatching
from models.bert import Bert
from models.matching_rep import MatchingRepresentation
from utils.constants import BERT, CONCEPTS
from utils.data_utils import load_source_sets, load_generations_sets, get_intervention_pairs, load_edits_sets
from utils.model_utils import get_fine_tuned_sentiment_model
from utils.results_utils import calculate_cebab_score, make_dir
import os

# parameters
SEED = 42

splits_to_explain = ['test',
                     # 'validation'
                     ]

causal_match = False
bert_match = False
approx = False
generative = True
random = False

models_to_explain = [
    'distil_bert',
    'bert',
    'roberta'
]

path_results = 'cebab_score'
make_dir(path_results)
path_results = 'cebab_score_filtered_generation'
make_dir(path_results)

# load sets
generations = load_generations_sets()
sources = load_source_sets()
edits = load_edits_sets(seed=SEED)
df_validation = sources['validation']
df_test = sources['test']
matching_set = edits[f'expanded_matching_set_{SEED}']

df_validation = sources['validation']
df_test = sources['test']
matching_set = edits[f'expanded_matching_set_{SEED}']

pairs_per_split = {split: get_intervention_pairs(df=sources[split], dataset_type="5-way", verbose=1) for split in
                   splits_to_explain}

# load explainers
explainers = []
if causal_match:
    tokenizer = AutoTokenizer.from_pretrained(BERT, return_tensors='pt')
    dir_path = 'saved_models/causal_models/'
    path_per_concept = {concept: os.path.join(dir_path, concept, 'model.pt') for concept in CONCEPTS}
    model_per_concept = {
        concept: MatchingRepresentation(pretrained_model_path=path_per_concept[concept], tokenizer=tokenizer) for
        concept in
        CONCEPTS}
    for concept in CONCEPTS:
        matching_set[f'{concept}_embeddings'] = model_per_concept[concept].get_embeddings(
            list(matching_set['text'].values))
    for k in range(1, 50):
        causal_matching_explainer = Matching(approach=1, set_to_match=matching_set,
                                             representation_model_per_concept=model_per_concept, assign=False, top_k=k,
                                             threshold=0, description=f'CausalMatching_k={k}')
        explainers.append(causal_matching_explainer)

if bert_match:
    bert = Bert()
    bert_matching_explainer = Matching(approach=1, set_to_match=matching_set,
                                       representation_model=bert, assign=True, top_k=1,
                                       threshold=0, description='bert_k=1')
    explainers.append(bert_matching_explainer)
    bert_matching_explainer = Matching(approach=1, set_to_match=matching_set,
                                       representation_model=bert, assign=True, top_k=2,
                                       threshold=0, description='bert_k=2')
    explainers.append(bert_matching_explainer)

if approx:
    approx_explainer = Approx(set_to_match=matching_set)
    explainers.append(approx_explainer)

if generative:
    generative_explainer = Generative(approach=1, filter_set=True)
    explainers.append(generative_explainer)

if random:
    random_explainer = RandomMatching(set_to_match=matching_set)
    explainers.append(random_explainer)

for model_to_explain in tqdm(models_to_explain):
    # load models to explain
    model = get_fine_tuned_sentiment_model(model_to_explain)

    model_path_results = os.path.join(path_results, model_to_explain)
    make_dir(model_path_results)
    # calculate cebab score

    for pair in pairs_per_split.keys():
        pairs = pairs_per_split[pair]
        split_to_explain = pair
        calculate_cebab_score(model_to_explain=model, concepts=CONCEPTS, explainers=explainers, pairs=pairs,
                              name=split_to_explain, wandb_on=False, path_dir=model_path_results, return_log=False,
                              save_outputs=False)
