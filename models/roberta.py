from models.classifier import Classifier
from transformers import RobertaTokenizer, \
    RobertaForSequenceClassification

from utils.constants import ROBERTA


class Roberta(Classifier):
    def __init__(self, pretrained_model_path=ROBERTA, num_labels=5):
        self.num_labels = num_labels
        self.pretrained_model_path = pretrained_model_path
        super().__init__(pretrained_model_path=self.pretrained_model_path, num_labels=self.num_labels)

    def get_representation_model(self):
        self.classifier.roberta

    def get_tokenizer(self):
        return RobertaTokenizer.from_pretrained(ROBERTA)

    def get_classifier(self):
        classifier = RobertaForSequenceClassification.from_pretrained(self.pretrained_model_path,
                                                                      num_labels=self.num_labels)
        return classifier
