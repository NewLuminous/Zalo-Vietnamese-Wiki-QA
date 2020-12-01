from .logit import Logit
from .logit_embedding import LogitWithEmbedding
from .crnn import CRNN
from .crnn_attention import AttentionCRNN
from .navie_bayes import navieBayes
from .rocchio import Rocchio
from .knn import KNN
from .svc import SVC_
from .random_forest import randomForest
from .extra_trees import ExtraTrees
from .lightgbm import LightGBM
from .xgboost import XGBoost

MODELS = [
    'navie-bayes',
    'rocchio', 'knn',
    'logit', 'logit-embedding',
    'svc',
    'random-forest', 'extra_trees', 'lightgbm', 'xgboost',
    'crnn', 'crnn-attention',
]

def get_model(model='logit'):
    if model == 'navie-bayes':
        return navieBayes
    elif model == 'rocchio':
        return Rocchio
    elif model == 'knn':
        return KNN
    elif model == 'logit':
        return Logit
    elif model == 'logit-embedding':
        return LogitWithEmbedding
    elif model == 'svc':
        return SVC_
    elif model == 'random-forest':
        return randomForest
    elif model == 'extra_trees':
        return ExtraTrees
    elif model == 'lightgbm':
        return LightGBM
    elif model == 'xgboost':
        return XGBoost
    elif model == 'crnn':
        return CRNN
    elif model == 'crnn-attention':
        return AttentionCRNN
    else:
        raise Exception(f"Model '{model}' not found. Try 'modeling.MODELS' for available models.")