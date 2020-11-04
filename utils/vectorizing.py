from utils import tokenizing
from utils import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

VECTORIZERS = ['tfidf']

def get_vectorizer(vectorizer='tfidf'):
    if vectorizer == 'tfidf':
        return TfidfVectorizer(tokenizer=tokenizing.tokenize,
                              preprocessor=preprocessing.preprocess)