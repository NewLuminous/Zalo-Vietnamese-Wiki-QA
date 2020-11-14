from .logit import Logit
from .logit_embedding import LogitWithEmbedding
from .crnn import CRNN
from .crnn_attention import AttentionCRNN

MODELS = ['logit', 'logit-embedding', 'crnn', 'crnn-attention']

def get_model(model='logit'):
    if model == 'logit':
        return Logit
    elif model == 'logit-embedding':
        return LogitWithEmbedding
    elif model == 'crnn':
        return CRNN
    elif model == 'crnn-attention':
        return AttentionCRNN
    else:
        raise Exception(f"Model '{model}' not found. Try 'modeling.MODELS' for available models.")