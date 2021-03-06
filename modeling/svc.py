import feature_extraction
import transforming
import numpy as np
import pandas as pd
from sklearn import svm


class SVC_:
    def __init__(self, vectorizer='tfidf', kernel='rbf', degree=3):
        self.vectorizer = feature_extraction.get(vectorizer)
        self.classifier = self.build_classifier(kernel, degree)

    def build_classifier(self, kernel, degree):
        return svm.SVC(kernel=kernel, degree=degree, decision_function_shape='ovr',
                       C=1, coef0=0.0, gamma=0.01, max_iter=-1, tol=0.001, cache_size=200,
                       class_weight=None, probability=True, random_state=None, shrinking=True, verbose=False)

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