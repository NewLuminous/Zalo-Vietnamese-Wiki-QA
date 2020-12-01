from .logit import Logit
from .logit_embedding import LogitWithEmbedding
from .crnn import CRNN
from .crnn_attention import AttentionCRNN
from .naive_bayes import naiveBayes
from .rocchio import Rocchio
from .knn import KNN
from .svc import SVC_
from .linear_svc import LinearSVC_
from .random_forest import randomForest
from .extra_trees import ExtraTrees
from .lightgbm import LightGBM
from .xgboost import XGBoost
from .bagging import Bagging

MODELS = [
    'naive_bayes',
    'rocchio', 'knn',
    'logit', 'logit-embedding',
    'svc', 'linear_svc',
    'random-forest', 'extra_trees', 'lightgbm', 'xgboost',
    'bagging',
    'crnn', 'crnn-attention',
]

def get_model(model='logit'):
    if model == 'naive_bayes':
        return naiveBayes
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
    elif model == 'linear_svc':
        return LinearSVC_
    elif model == 'random-forest':
        return randomForest
    elif model == 'extra_trees':
        return ExtraTrees
    elif model == 'lightgbm':
        return LightGBM
    elif model == 'xgboost':
        return XGBoost
    elif model == 'bagging':
        return Bagging
    elif model == 'crnn':
        return CRNN
    elif model == 'crnn-attention':
        return AttentionCRNN
    else:
        raise Exception(f"Model '{model}' not found. Try 'modeling.MODELS' for available models.")