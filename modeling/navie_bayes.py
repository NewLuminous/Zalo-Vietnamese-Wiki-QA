import feature_extraction
import transforming
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

class navieBayes:
    def __init__(self, vectorizer='tfidf'):
        self.vectorizer = feature_extraction.get(vectorizer)
        self.classifier = self.build_classifier()

    def build_classifier(self):
        return GaussianNB()

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


