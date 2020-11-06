import vectorizing
import transforming
import numpy as np
from sklearn.linear_model import LogisticRegression

class Logit:
    def __init__(self, vectorizer='tfidf', random_state=None):
        self.vectorizer = vectorizing.get_vectorizer(vectorizer)
        self.model = self.build_model(random_state)

    def build_model(self, random_state):
        return LogisticRegression(random_state=random_state)
        
    def transform_using_word2vec(self, X):
        self.vectorizer.fit(X['question'])
        self.vectorizer.fit(X['answer'])
        X_question = np.vstack(self.vectorizer.transform(X['question'], minlen=30))
        X_answer = np.vstack(self.vectorizer.transform(X['answer'], minlen=500))
        return np.hstack([X_question, X_answer])

    def fit(self, X, y):
        if type(self.vectorizer) is vectorizing.Word2Vec:
            X = self.transform_using_word2vec(X)
        else:
            X = transforming.concatenate_after_vectorizing(X, self.vectorizer)
        self.model.fit(X, y)

    def predict(self, X):
        if type(self.vectorizer) is vectorizing.Word2Vec:
            X = self.transform_using_word2vec(X)
        else:
            X = transforming.concatenate_after_vectorizing(X, self.vectorizer, do_fit_vectorizer=False)
        return self.model.predict(X)

    def predict_proba(self, X):
        if type(self.vectorizer) is vectorizing.Word2Vec:
            X = self.transform_using_word2vec(X)
        else:
            X = transforming.concatenate_after_vectorizing(X, self.vectorizer, do_fit_vectorizer=False)
        return self.model.predict_proba(X)