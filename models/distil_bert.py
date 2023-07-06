from models.classifier import Classifier
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from utils.constants import BERT


class Distil_Bert(Classifier):
    def __init__(self, pretrained_model_path="distilbert-base-uncased", num_labels=5):
        self.num_labels = num_labels
        self.pretrained_model_path = pretrained_model_path
        super().__init__(pretrained_model_path=self.pretrained_model_path, num_labels=self.num_labels)

    def get_representation_model(self):
        return self.classifier.bert

    def get_tokenizer(self):
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", return_tensors='pt')
        return tokenizer

    def get_classifier(self):
        model = DistilBertForSequenceClassification.from_pretrained(self.pretrained_model_path,
                                                                    num_labels=self.num_labels)

        return model
