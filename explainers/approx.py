import numpy as np

from explainers.matching_based_explainer import MatchingBasedExplainer
from utils.constants import CONCEPTS
from utils.metric_utils import cosine_similarity_matrix


class Approx(MatchingBasedExplainer):

    def __init__(self, set_to_match, representation_model=None, description=None, use_aspects_labels=False):
        super().__init__(set_to_match=set_to_match, representation_model=representation_model, description=description)
        self.set_to_match = self.set_to_match.dropna(subset=['text'])
        self.approach = 'sample_approx'
        if self.representation_model is not None:
            self.approach = 'exact_approx'
            self.set_to_match['embeddings'] = self.representation_model.get_embeddings(
                list(self.set_to_match['text'].values))

    def icace_error(self, model, pairs, concept, base_direction, target_direction, save_outputs=False):
        # aspects = [f'{c}_aspect_majority' for c in CONCEPTS if c != concept]
        pairs = pairs.copy()
        pairs = pairs.reset_index()
        encode_str = {'tensor(0)': 'unknown', 'tensor(1)': 'Negative', 'tensor(2)': 'Positive'}
        for c in CONCEPTS:
            pairs[f'{c}_predictions_base'] = pairs[f'{c}_predictions_base'].apply(lambda x: encode_str[str(x)])

        def compute_icace(row):
            original_id = row['original_id_base']
            candidates = self.set_to_match[self.set_to_match[f'{concept}_label'] == target_direction]
            for c in CONCEPTS:
                if c != concept:
                    candidates = candidates[candidates[f'{c}_predictions'] == row[f'{c}_predictions_base']]

            if len(candidates) == 0:
                return None
            if self.approach == 'sample_approx':
                match = candidates.sample(n=1)
                row['prediction_match'] = model.get_predictions(list(match[f'text'].values))[0]
            elif self.approach == 'exact_approx':
                match_idx = self.find_match_idx(candidates, row)
                match = candidates.loc[match_idx]
                row['prediction_match'] = model.get_predictions(match[f'text'])
            else:
                raise Exception('Invalid approach')

            explanation = np.array(row[f'prediction_match']) - np.array(
                row[f'prediction_base'])
            icace = np.array(row[f'prediction_counterfactual']) - np.array(
                row[f'prediction_base'])
            return np.linalg.norm(explanation - icace, ord=2)

        pairs['icace_error'] = pairs.apply(lambda row: compute_icace(row), axis=1)
        if save_outputs:
            self.save_matches(pairs, concept, base_direction, target_direction)
        return pairs

    def get_explainer_description(self):
        if self.representation_model is None:
            return 'approx'
        return 'approxiMatch'

    def find_match_idx(self, candidates, example):
        indexes = candidates.index
        if len(indexes) == 1:
            return indexes.values[0]
        candidates = candidates.reset_index()

        candidates_embeddings = candidates['embeddings'].values
        # embeddings bases
        bases_embeddings = self.representation_model.get_embeddings([example['description_base']])

        dist_mat = cosine_similarity_matrix(bases_embeddings, candidates_embeddings)

        match_temp_idx = list(np.argmax(dist_mat, axis=1))
        match_original_idx = indexes[match_temp_idx].values[0]
        return match_original_idx

    def fit(self):
        pass

    def set_representation_model(self, model):
        self.representation_model = model
        if self.approach == 'exact_approx':
            self.set_to_match['embeddings'] = self.representation_model.get_embeddings(
                list(self.set_to_match['text'].values))
