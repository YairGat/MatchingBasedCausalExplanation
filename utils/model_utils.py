from models.distil_bert import Distil_Bert
from models.roberta import Roberta
from utils.constants import BERT_SENTIMENT_PATH, DISTIL_BERT_SENTIMENT_PATH, ROBERTA_SENTIMENT_PATH
from models.bert import Bert


def get_fine_tuned_sentiment_model(architecture):
    if architecture == 'bert':
        model_path = BERT_SENTIMENT_PATH
        model = Bert(pretrained_model_path=model_path)
        return model
    elif architecture == 'distil_bert':
        model_path = DISTIL_BERT_SENTIMENT_PATH
        model = Distil_Bert(pretrained_model_path=model_path)
        return model
    elif architecture == 'roberta':
        model_path = ROBERTA_SENTIMENT_PATH
        model = Roberta(pretrained_model_path=model_path)
        return model
    elif architecture == 'gpt3':
        raise NotImplementedError
    else:
        raise NotImplementedError
