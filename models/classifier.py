from abc import ABC, abstractmethod
import torch
from math import ceil
import numpy as np
from models.model import Model
import os
from transformers import Trainer, DataCollatorWithPadding
from utils.results_utils import make_dir
from utils.constants import SAVED_TRAINED_MODELS_PATH
from utils.constants import DEVICE
from utils.training_utils import tokenization_for_training
import evaluate


class Classifier(Model, ABC):
    def __init__(self, pretrained_model_path, num_labels):
        self.num_labels = num_labels
        self.classifier = self.get_classifier()
        super().__init__(pretrained_model_path=pretrained_model_path)

    def train(self, aspect, training_args, tokenized_train, tokenized_valid, tokenized_test=None, path_to_save=None):
        def compute_metrics(eval_pred):
            metric = evaluate.load("accuracy")
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        trainer = Trainer(model=self.classifier,
                          args=training_args,
                          train_dataset=tokenized_train,
                          eval_dataset=tokenized_valid,
                          compute_metrics=compute_metrics,
                          # tokenizer=self.tokenizer,
                          # data_collator=DataCollatorWithPadding(self.tokenizer)
                          )
        # trainer = Trainer(model=self.classifier,
        #                   args=training_args,
        #                   train_dataset=tokenized_train,
        #                   eval_dataset=tokenized_valid,
        #                   compute_metrics=compute_metrics,
        #                   tokenizer=self.tokenizer
        #                   )
        trainer.train()

        self.classifier = trainer.model
        if path_to_save is not None:
            make_dir(SAVED_TRAINED_MODELS_PATH)
            path = os.path.join(SAVED_TRAINED_MODELS_PATH, path_to_save)
            make_dir(path)
            path_model = os.path.join(path, f'{self.get_model_description()}')
            trainer.model.save_pretrained(path_model)

            if len(tokenized_valid) > 0:
                valid_accuracy = trainer.predict(tokenized_valid).metrics
                f = open(os.path.join(path, f'validation_evaluation.txt'), 'w')
                f.write(str(valid_accuracy))

            if tokenized_test is not None:
                test_accuracy = trainer.predict(tokenized_test).metrics
                f = open(os.path.join(path, f'test_evaluation.txt'), 'w')
                f.write(str(test_accuracy))

            train_accuracy = trainer.predict(tokenized_train).metrics
            f = open(os.path.join(path, f'train_evaluation.txt'), 'w')
            f.write(str(train_accuracy))

    def get_predictions(self, lst_texts, return_predictions=False, batch_size=-1):
        if batch_size != -1:
            self.batch_size = batch_size
        tokenized_set = self.tokenizer(lst_texts, truncation=True, return_tensors='pt', padding=True)
        self.classifier.to(DEVICE)
        self.classifier.eval()
        # get the predictions batch per batch
        probas = []
        preds = []
        for i in range(ceil(len(tokenized_set['input_ids']) / self.batch_size)):
            x_batch = {k: v[i * self.batch_size:(i + 1) * self.batch_size].to(DEVICE) for k, v in
                       tokenized_set.items()}
            with torch.no_grad():
                outputs = torch.nn.functional.softmax(self.classifier(**x_batch).logits.detach().cpu(), dim=-1)
                probas += list(outputs.tolist())
                preds += list(torch.argmax(outputs, dim=1))
            # print('memory cleaning')
            del x_batch
            torch.cuda.empty_cache()
        self.classifier = self.classifier.cpu()
        if return_predictions:
            return preds

        # tokenized_set.to('cpu')
        return probas

    @abstractmethod
    def get_classifier(self):
        raise NotImplemented

    def tokenize_sets(self, sets, label_column='review_majority'):
        return tokenization_for_training(splits=sets, tokenizer=self.tokenizer,
                                         label_column=label_column)

    def get_num_labels(self):
        return self.num_labels
