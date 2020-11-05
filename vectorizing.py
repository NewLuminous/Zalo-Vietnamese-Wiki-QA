import tokenizing
import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

VECTORIZERS = ['count', 'tfidf']

def get_vectorizer(vectorizer='tfidf'):
    if vectorizer == 'count':
        return CountVectorizer(tokenizer=tokenizing.tokenize,
                              preprocessor=preprocessing.preprocess)
    elif vectorizer == 'tfidf':
        return TfidfVectorizer(tokenizer=tokenizing.tokenize,
                              preprocessor=preprocessing.preprocess)