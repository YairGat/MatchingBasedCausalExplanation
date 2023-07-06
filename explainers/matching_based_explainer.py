import os
from abc import ABC, abstractmethod

import pandas as pd

from explainers.explainer import Explainer
from utils.constants import RESULTS_PATH
from utils.results_utils import make_dir


class MatchingBasedExplainer(Explainer, ABC):
    def __init__(self, set_to_match, description, representation_model=None, representation_model_per_concept=None):
        super().__init__()
        self.set_to_match = set_to_match
        self.description = description
        self.representation_model = representation_model
        self.representation_model_per_concept = representation_model_per_concept

    @abstractmethod
    def set_representation_model(self, model):
        raise NotImplemented()

    def save_matches(self, all_pairs, concept, base_direction, target_direction):
        p = os.path.join(RESULTS_PATH, 'matches_icace')
        make_dir(p)
        p = os.path.join(p, f'{base_direction}->{target_direction}')
        make_dir(p)
        p = os.path.join(p, f'{self.get_explainer_description()}.csv')
        df = pd.DataFrame()
        df['factual'] = list(all_pairs['description_base'].values)
        matching_columns = [c for c in all_pairs.columns if 'description_match' in c]
        for c in matching_columns:
            df[c] = list(all_pairs[c].values)
        df['counterfactual'] = list(all_pairs['description_counterfactual'].values)
        df['icace'] = list(all_pairs['icace_error'].values)
        df.to_csv(p, index=False)
