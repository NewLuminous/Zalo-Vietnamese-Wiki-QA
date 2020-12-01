import feature_extraction
import transforming
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

class LightGBM:
    def __init__(self, vectorizer='tfidf', num_leaves=23, learning_rate=0.01, max_depth=-1, num_boost_round=99999):
        self.vectorizer = feature_extraction.get(vectorizer)
        self.classifier = self.build_classifier(num_leaves, learning_rate, max_depth, num_boost_round)

    def build_classifier(self, num_leaves, learning_rate, max_depth, num_boost_round):
        return LGBMClassifier(num_leaves=num_leaves, learning_rate=learning_rate, max_depth=max_depth, num_boost_round=num_boost_round,
                              min_split_gain=0.007, feature_fraction=0.106, bagging_fraction=0.825,
                              lambda_l1=0.2, lambda_l2=2.7, objective='binary')

    def fit(self, X, y, eval_set=None):
        X = transforming.vectorize_and_concatenate_qa(X, self.vectorizer)
        if eval_set is not None:
            X_val = transforming.vectorize_and_concatenate_qa(eval_set[0], self.vectorizer, do_fit_vectorizer=False)
            eval_set = (X_val, eval_set[1])
            
        self.classifier.fit(X, y, eval_set=eval_set, early_stopping_rounds = 800, verbose = 100)

    def predict(self, X):
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(np.reshape(X, (-1, 2)), columns=['question', 'text'])
    
        X = transforming.vectorize_and_concatenate_qa(X, self.vectorizer, do_fit_vectorizer=False)
        return self.classifier.predict(X)

    def predict_proba(self, X):
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(np.reshape(X, (-1, 2)), columns=['question', 'text'])
            
        X = transforming.vectorize_and_concatenate_qa(X, self.vectorizer, do_fit_vectorizer=False)
        return self.classifier.predict_proba(X)