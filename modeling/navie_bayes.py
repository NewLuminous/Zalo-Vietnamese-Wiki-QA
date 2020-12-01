import feature_extraction
import transforming
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

class navieBayes:
    def __init__(self, vectorizer='tfidf', alpha = 1.0):
        self.vectorizer = feature_extraction.get(vectorizer)
        self.classifier = self.build_classifier(alpha)

    def build_classifier(self, alpha):
        return MultinomialNB(alpha = alpha)

    def fit(self, X, y):
        X = transforming.vectorize_and_concatenate_qa(X, self.vectorizer)
        self.classifier.fit(X, y)

    def predict(self, X):
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(np.reshape(X, (-1, 2)), columns=['question', 'text'])

        X = transforming.vectorize_and_concatenate_qa(X, self.vectorizer, do_fit_vectorizer=False)
        return self.classifier.predict(X)

    def get_classifier_params(self):
        return self.classifier.get_params()

    def grid_search(self, X, y, param_grid, scoring=None, n_jobs=-2):
        X = transforming.vectorize_and_concatenate_qa(X, self.vectorizer)
        gs = GridSearchCV(estimator=self.classifier, param_grid=param_grid, scoring=scoring, cv=3, verbose=1,
                          n_jobs=n_jobs)
        gs.fit(X, y)
        return gs.best_params_

    def predict_proba(self, X):
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(np.reshape(X, (-1, 2)), columns=['question', 'text'])

        X = transforming.vectorize_and_concatenate_qa(X, self.vectorizer, do_fit_vectorizer=False)
        return self.classifier.predict_proba(X)


