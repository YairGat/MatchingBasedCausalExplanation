from models.classifier import Classifier
from transformers import AutoTokenizer, BertForSequenceClassification
from utils.constants import BERT


class Bert(Classifier):
    def __init__(self, pretrained_model_path=BERT, num_labels=5):
        self.num_labels = num_labels
        self.pretrained_model_path = pretrained_model_path
        super().__init__(pretrained_model_path=self.pretrained_model_path, num_labels=self.num_labels)

    def get_representation_model(self):
        return self.classifier.bert

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(BERT, return_tensors='pt')

    def get_classifier(self):
        classifier = BertForSequenceClassification.from_pretrained(self.pretrained_model_path, num_labels=self.num_labels)
        return classifier

