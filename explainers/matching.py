import numpy as np

from explainers.matching_based_explainer import MatchingBasedExplainer
from utils.metric_utils import cosine_similarity_matrix, cosine_similarity_matrix_2


class Matching(MatchingBasedExplainer):
    def __init__(self, set_to_match, approach, representation_model=None, top_k=1, threshold=0.5, description=None,
                 assign=False, representation_model_per_concept=None):
        super().__init__(set_to_match=set_to_match, description=description, representation_model=representation_model,
                         representation_model_per_concept=representation_model_per_concept)
        self.top_k = top_k
        self.threshold = threshold
        self.approach = approach
        self.set_to_match = self.set_to_match.dropna(subset=['text'])
        if assign and representation_model is not None:
            self.set_to_match['embeddings'] = self.representation_model.get_embeddings(
                list(self.set_to_match['text'].values))
        if assign and representation_model_per_concept is not None:
            for concept in self.representation_model_per_concept.keys():
                self.set_to_match[f'{concept}_embeddings'] = self.representation_model_per_concept[
                    concept].get_embeddings(list(self.set_to_match['text'].values))

    def icace_error(self, model, pairs, concept, base_direction, target_direction, save_outputs=False):
        pairs = pairs.copy()
        pairs = pairs.reset_index()
        candidates = self.set_to_match[self.set_to_match[f'{concept}_label'] ==
                                       target_direction]
        # drop rows with a nan text
        # candidates = candidates.dropna(subset=['text'])
        candidates = candidates.reset_index()
        if self.representation_model_per_concept is not None:
            candidates_embeddings = candidates[f'{concept}_embeddings']
            bases_embeddings = self.representation_model_per_concept[concept].get_embeddings(
                list(pairs['description_base'].values))
        elif self.representation_model is not None:
            # embedding candidates
            candidates_embeddings = candidates['embeddings']
            # embeddings bases
            bases_embeddings = self.representation_model.get_embeddings(
                list(pairs['description_base'].values))
        else:
            raise Exception('No representation model found')
        dist_mat = cosine_similarity_matrix_2(bases_embeddings, candidates_embeddings)
        # dist_mat = cosine_similarity_matrix(bases_embeddings, candidates_embeddings)

        matches_indexes = np.argsort(-dist_mat, axis=1)[:, :self.top_k]
        similarities_values = np.take_along_axis(dist_mat, matches_indexes, axis=1)

        # pairs[f'prediction_base'] = model.get_predictions(list(pairs['description_base'].values))

        for k in range(self.top_k):
            pairs[f'description_match_{k}'] = candidates.loc[matches_indexes[:, k]]['text'].values
            pairs[f'prediction_match_{k}'] = model.get_predictions(list(pairs[f'description_match_{k}'].values))
            pairs[f'similarity_match_{k}'] = similarities_values[:, k]

        # pairs[f'prediction_counterfactual'] = model.get_predictions(list(pairs['description_counterfactual'].values))

        def icace_error_approach_1(row):
            under_the_threshold = 0
            prediction_base = row['prediction_base']
            prediction_counterfactual = row['prediction_counterfactual']

            s = np.array([0] * len(prediction_base))
            for i in range(self.top_k):
                if row[f'similarity_match_{i}'] > self.threshold:
                    s = s + np.array(row[f'prediction_match_{i}'])
                    under_the_threshold += 1
            if under_the_threshold == 0:
                return None
            else:
                explanation = s / under_the_threshold - np.array(prediction_base)
                icace = np.array(prediction_counterfactual) - np.array(prediction_base)

                return np.linalg.norm(explanation - icace, ord=2)

        def icace_error_approach_2(row):
            under_the_threshold = 0
            prediction_base = row['prediction_base']
            explanations = []
            for i in range(self.top_k):
                if row[f'similarity_match_{i}'] > self.threshold:
                    explanations.append(np.array(row[f'prediction_match_{i}']) - np.array(prediction_base))
                    under_the_threshold += 1
            if under_the_threshold == 0:
                return None
            else:
                icace = np.array(row['prediction_counterfactual']) - np.array(prediction_base)
                s = 0
                for i in range(under_the_threshold):
                    s += np.linalg.norm(np.array(explanations[i]) - np.array(icace), ord=2)

                return s / under_the_threshold

        if self.approach == 1:
            pairs['icace_error'] = pairs.apply(lambda row: icace_error_approach_1(row), axis=1)
        elif self.approach == 2:
            pairs['icace_error'] = pairs.apply(lambda row: icace_error_approach_2(row), axis=1)
        else:
            raise ValueError('approach must be 1 or 2')

        # drop None values of icace error
        pairs = pairs.dropna(subset=['icace_error'])
        if save_outputs:
            self.save_matches(pairs, concept, base_direction, target_direction)
        return pairs

    def get_explainer_description(self):
        if self.description is not None:
            return self.description
        k = ''
        if self.top_k != 1:
            k = f'_k={self.top_k}'
        approach = ''
        if self.approach == 2:
            approach = '2'
        threshold = ''
        if self.threshold != 0:
            threshold = f'_t={self.threshold}'
        return f'Matching{approach}{k}{threshold}'

    def fit(self):
        raise NotImplemented

    def set_representation_model(self, model):
        self.representation_model = model
        self.set_to_match['embeddings'] = self.representation_model.get_embeddings(
            list(self.set_to_match['text'].values))
        print(f'\nlm is changed and embeddings are updated for the explainer {self.get_explainer_description()}')

    def set_representation_model_per_concept(self, model, concepts):
        for concept in concepts:
            self.representation_model_per_concept[concept] = model
            self.set_to_match[f'{concept}_embeddings'] = self.representation_model_per_concept[concept].get_embeddings(
                list(self.set_to_match['text'].values))
        print(
            f'\nlm is changed and embeddings are updated for the explainer {self.get_explainer_description()}, for concepts {concepts}')
