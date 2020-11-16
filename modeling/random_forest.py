import feature_extraction
import transforming
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class randomForest:
    def __init__(self, vectorizer='tfidf', max_depth = 10, random_state=0):
        self.vectorizer = feature_extraction.get(vectorizer)
        self.classifier = self.build_classifier(max_depth, random_state)

    def build_classifier(self, max_depth, random_state):
        return RandomForestClassifier(max_depth=max_depth, random_state = random_state)

    def fit(self, X, y):
        X = transforming.vectorize_and_concatenate_qa(X, self.vectorizer)
        self.classifier.fit(X, y)

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