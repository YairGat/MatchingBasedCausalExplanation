import numpy as np

from explainers.matching_based_explainer import MatchingBasedExplainer


class RandomMatching(MatchingBasedExplainer):
    def __init__(self, set_to_match, description=None):
        super().__init__(set_to_match=set_to_match, description=description)

    def icace_error(self, model, pairs, concept, base_direction, target_direction, save_outputs=False):
        pairs = pairs.copy()
        pairs = pairs.reset_index()
        candidates = self.set_to_match[self.set_to_match[f'{concept}_label'] ==
                                       target_direction]
        candidates = candidates.reset_index()
        # embedding candidates
        matches = candidates.sample(n=len(pairs))
        pairs['description_match'] = matches['text'].values
        pairs[f'prediction_match'] = model.get_predictions(list(matches['text'].values))

        def compute_icace(row):
            explanation = np.array(row[f'prediction_match']) - np.array(row[f'prediction_base'])
            icace = np.array(row[f'prediction_counterfactual']) - np.array(row[f'prediction_base'])
            return np.linalg.norm(explanation - icace, ord=2)

        pairs['icace_error'] = pairs.apply(lambda row: compute_icace(row), axis=1)
        if save_outputs:
            self.save_matches(pairs, concept, base_direction, target_direction)
        return pairs

    def get_explainer_description(self):
        return f'random matching'

    def fit(self):
        raise NotImplemented

    def set_representation_model(self, model):
        print(f'no model in {self.get_explainer_description}')
