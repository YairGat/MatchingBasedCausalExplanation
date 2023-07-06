from abc import ABC, abstractmethod
import torch
from math import ceil
import numpy as np
from utils.constants import DEVICE


class Model(ABC):

    def __init__(self, pretrained_model_path, batch_size=32, model_description=None):
        self.pretrained_model_path = pretrained_model_path
        self.model_description = model_description
        self.batch_size = batch_size
        self.tokenizer = self.get_tokenizer()

    @abstractmethod
    def get_tokenizer(self):
        raise NotImplemented

    @abstractmethod
    def get_representation_model(self):
        raise NotImplemented

    def train(self):
        raise NotImplemented

    def get_embeddings(self, lst_texts):
        tokenized_set = self.tokenizer(lst_texts, truncation=True, return_tensors='pt', padding=True)
        lm = self.get_representation_model()
        lm.to(DEVICE)
        lm.eval()
        # get the predictions batch per batch
        lst = []
        for i in range(ceil(len(tokenized_set['input_ids']) / self.batch_size)):
            x_batch = {k: v[i * self.batch_size:(i + 1) * self.batch_size].to(DEVICE) for k, v in
                       tokenized_set.items()}
            with torch.no_grad():
                lst += lm(**x_batch).pooler_output.detach().cpu().tolist()
            del x_batch
            torch.cuda.empty_cache()

        np_embeddings = [np.array(e) for e in lst]
        lm.cpu()
        return np_embeddings

    def get_model_description(self):
        if self.model_description is None:
            return f'{self.pretrained_model_path}'
        else:
            return f'{self.model_description}'
