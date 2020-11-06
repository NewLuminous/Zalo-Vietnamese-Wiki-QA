from .logit import Logit
from .crnn import CRNN

MODELS = ['logit', 'crnn']

def get_model(model='logit'):
    if model == 'logit':
        return Logit
    elif model == 'crnn':
        return CRNN