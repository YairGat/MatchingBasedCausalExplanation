from transformers import GPT2Tokenizer, GPT2Model, GPT2ForSequenceClassification
from models.classifier import Classifier
import torch.nn as nn

from utils.training_utils import tokenization_for_training_gpt


class GPT2(Classifier):
    def __init__(self, pretrained_model_path='gpt2-large', num_labels=5):
        self.num_labels = num_labels
        self.pretrained_model_path = pretrained_model_path
        super().__init__(pretrained_model_path=self.pretrained_model_path, num_labels=self.num_labels)

    def get_representation_model(self):
        return self.classifier.bert

    def get_tokenizer(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large', padding=True, truncation=True, return_tensors='pt',
                                                       max_length=512)
        self.tokenizer.add_tokens('[PAD]')
        self.tokenizer.pad_token = '[PAD]'
        self.tokenizer.pad_token_id = self.tokenizer.encode('[PAD]')[0]

        return self.tokenizer

    def get_classifier(self):
        # add a classification head to self.gpt_2
        self.classifier = GPT2ForSequenceClassification.from_pretrained(self.pretrained_model_path,
                                                                        num_labels=self.num_labels)
        self.classifier.resize_token_embeddings(len(self.get_tokenizer()))

        return self.classifier

#
# # Define the complete model
# class GPT2Classifier(nn.Module):
#     def __init__(self, num_classes):
#         super(GPT2Classifier, self).__init__()
#
#         # Load the GPT2 model
#         self.gpt2 = GPT2Model.from_pretrained('gpt2-large')
#
#         # Get the hidden size of the GPT2 model
#         hidden_size = self.gpt2.config.hidden_size
#
#         # Define the linear classification head
#         self.classification_head = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, input_ids, attention_mask):
#         # Forward pass through GPT2 model
#         outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
#
#         # Get the last hidden state from GPT2 model outputs
#         last_hidden_state = outputs.last_hidden_state
#
#         # Apply linear classification head
#         logits = self.classification_head(last_hidden_state[:, 0, :])
#
#         return logits
