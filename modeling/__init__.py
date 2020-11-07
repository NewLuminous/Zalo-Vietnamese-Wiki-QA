from .logit import Logit
from .crnn import CRNN
from .crnn_attention import AttentionCRNN

MODELS = ['logit', 'crnn']

def get_model(model='logit'):
    if model == 'logit':
        return Logit
    elif model == 'crnn':
        return CRNN
    elif model == 'crnn-attention':
        return AttentionCRNN