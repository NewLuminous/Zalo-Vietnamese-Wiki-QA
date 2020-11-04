from .logit import Logit

MODELS = ['logit']

def get_model(model='logit'):
    if model == 'logit':
        return Logit