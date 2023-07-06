from explainers.matching_based_explainer import MatchingBasedExplainer
import pandas as pd
import numpy as np


class Generative(MatchingBasedExplainer):

    def __init__(self, approach, description=None, filter_set=False):
        if filter_set:
            set_to_match = pd.read_csv('sets/filtered_generations/all_generations.csv')
        else:
            set_to_match = pd.read_csv('sets/generations/all_generations.csv')

        # for this explainer the set to match is a set of generations.\
        self.approach = approach
        super().__init__(set_to_match=set_to_match, description=description)
        self.set_to_match = self.set_to_match.dropna(subset=['generation'])

    def icace_error(self, model, pairs, concept, base_direction, target_direction, save_outputs=False):
        relevant_generations = self.set_to_match[(self.set_to_match[f'intervention_aspect'] == concept) & (
                self.set_to_match[f'target_direction'] == target_direction)]
        pairs = pairs.copy()
        pairs = pairs.reset_index()

        # pairs[f'prediction_base'] = model.get_predictions(list(pairs['description_base'].values))
        # pairs[f'prediction_counterfactual'] = model.get_predictions(list(pairs['description_counterfactual'].values))
        def icace_error_approach_1(row):
            original_id = row['original_id_base']
            idx_row = row['index']
            generations_slice = relevant_generations[relevant_generations['original_id'] == original_id]
            if len(generations_slice) == 0:
                return None

            generations_slice['prediction'] = model.get_predictions(list(generations_slice['generation'].values))
            s = np.array([0] * len(row['prediction_base']))
            for i in range(len(generations_slice)):
                example = generations_slice.iloc[i]
                pairs.loc[idx_row, f'description_match_{i}'] = example['generation']
                s = s + np.array(example['prediction'])

            explanation = s / len(generations_slice) - np.array(row['prediction_base'])
            icace = np.array(row['prediction_counterfactual']) - np.array(row['prediction_base'])
            return np.linalg.norm(explanation - icace, ord=2)

        def icace_error_approach_2(row):

            original_id = row['original_id_base']
            idx_row = row['index']
            generations_slice = relevant_generations[relevant_generations['original_id'] == original_id]
            if generations_slice.empty:
                return None
            generation = generations_slice.sample(1)
            pairs.loc[idx_row, 'description_match'] = generation['generation'].values
            prediction_match = model.get_predictions(list(pairs.loc[idx_row, 'description_match']))
            icace = np.array(row['prediction_counterfactual']) - np.array(row['prediction_base'])
            explanation = np.array(prediction_match) - np.array(row['prediction_base'])

            return np.linalg.norm(explanation - icace, ord=2)

        if self.approach == 1:
            pairs['icace_error'] = pairs.apply(lambda x: icace_error_approach_1(x), axis=1)
        elif self.approach == 2:
            pairs['icace_error'] = pairs.apply(lambda x: icace_error_approach_2(x), axis=1)
        else:
            raise ValueError('approach should be 1 or 2')

        pairs = pairs.dropna(subset=['icace_error'])
        if save_outputs:
            self.save_matches(pairs, concept, base_direction, target_direction)
        return pairs

    def get_explainer_description(self):
        return 'generative'

    def fit(self):
        raise NotImplemented

    def set_representation_model(self, model):
        print(f'no model in {self.get_explainer_description}, app-{self.approach}')
