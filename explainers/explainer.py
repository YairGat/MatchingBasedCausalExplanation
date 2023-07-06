from abc import ABC, abstractmethod


class Explainer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_explainer_description(self):
        raise NotImplemented

    @abstractmethod
    def fit(self):
        raise NotImplemented

    @abstractmethod
    def icace_error(self, model, pairs, concept, base_direction, target_direction, save_outputs=False):
        raise NotImplemented()
