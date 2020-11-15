from .logit import Logit
from .logit_embedding import LogitWithEmbedding
from .crnn import CRNN
from .crnn_attention import AttentionCRNN
from .navie_bayes import navieBayes
from .knn import KNN
from .svc import SVC_

MODELS = ['logit', 'logit-embedding', 'crnn', 'crnn-attention', 'navie-bayes','knn','svc']

def get_model(model='logit'):
    if model == 'logit':
        return Logit
    elif model == 'logit-embedding':
        return LogitWithEmbedding
    elif model == 'crnn':
        return CRNN
    elif model == 'crnn-attention':
        return AttentionCRNN
    elif model == 'navie-bayes':
        return navieBayes
    elif model == 'knn':
        return KNN
    elif model == 'svc':
        return SVC_
    else:
        raise Exception(f"Model '{model}' not found. Try 'modeling.MODELS' for available models.")