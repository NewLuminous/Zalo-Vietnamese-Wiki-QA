import vectorizing
import transforming
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class Logit:
    def __init__(self, vectorizer='tfidf', random_state=None):
        self.vectorizer = vectorizing.get_vectorizer(vectorizer)
        self.model = self.build_model(random_state)

    def build_model(self, random_state):
        return LogisticRegression(random_state=random_state)

    def fit(self, X, y):
        X = transforming.vectorize_and_concatenate_qa(X, self.vectorizer)
        self.model.fit(X, y)

    def predict(self, X):
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(np.reshape(X, (-1, 2)), columns=['question', 'text'])
    
        X = transforming.vectorize_and_concatenate_qa(X, self.vectorizer, do_fit_vectorizer=False)
        return self.model.predict(X)

    def predict_proba(self, X):
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(np.reshape(X, (-1, 2)), columns=['question', 'text'])
            
        X = transforming.vectorize_and_concatenate_qa(X, self.vectorizer, do_fit_vectorizer=False)
        return self.model.predict_proba(X)