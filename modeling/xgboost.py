import feature_extraction
import transforming
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

class XGBoost:
    def __init__(self, vectorizer='tfidf', max_depth=8, tree_method='auto'):
        self.vectorizer = feature_extraction.get(vectorizer)
        self.classifier = self.build_classifier(max_depth, tree_method)

    def build_classifier(self, max_depth, tree_method):
        return XGBClassifier(max_depth=max_depth, tree_method=tree_method,
                             seed=1000, gamma=0.1, subsample=0.7, colsample_bytree=0.7, min_child_weight=3, eta=0.1,
                             booster="gbtree", objective="binary:logistic")

    def fit(self, X, y):
        X = transforming.vectorize_and_concatenate_qa(X, self.vectorizer).tocsr()
        self.classifier.fit(X, y, verbose = 200)

    def predict(self, X):
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(np.reshape(X, (-1, 2)), columns=['question', 'text'])
    
        X = transforming.vectorize_and_concatenate_qa(X, self.vectorizer, do_fit_vectorizer=False).tocsr()
        return self.classifier.predict(X)

    def predict_proba(self, X):
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(np.reshape(X, (-1, 2)), columns=['question', 'text'])
            
        X = transforming.vectorize_and_concatenate_qa(X, self.vectorizer, do_fit_vectorizer=False).tocsr()
        return self.classifier.predict_proba(X)