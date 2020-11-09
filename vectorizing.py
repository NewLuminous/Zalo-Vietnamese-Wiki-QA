import numpy as np
import tokenizing
import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim

VECTORIZERS = ['count', 'tfidf', 'word2vec']

class Word2Vec:
    def __init__(self, tokenizer, preprocessor):
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.model = None

    def fit(self, sentences):
        if self.model is None:
            #self.model = gensim.models.Word2Vec(sentences, min_count=10, size=50, workers=12, window=3, sg=1)
            self.model = gensim.models.Word2Vec.load("saved_models/word2vec_news.model")
        return self

    def vectorize_sentence(self, sentence, minlen=None, maxlen=None):
        sentence = self.tokenizer(self.preprocessor(sentence))
        last_unknown = False
        embs = []
        for token in sentence:
            if token in self.model.wv:
                embs.append(self.model.wv[token])
                last_unknown = False
            elif not last_unknown:
                #embs.append(np.zeros(self.model.vector_size))
                embs.append(self.model.wv['&'])
                last_unknown = True

        embs = np.array(embs)
        if minlen is not None and embs.shape[0] < minlen:
            paddings = np.ceil(minlen / embs.shape[0]) - 1
            d = np.copy(embs)
            for i in range(int(paddings)):
                embs = np.concatenate((embs, d))
            embs = embs[0: minlen]
                
        if maxlen is not None and embs.shape[0] > maxlen:
            embs = embs[0: maxlen]

        return embs

    def transform(self, sentences, minlen=None, maxlen=None):
        if self.model is None:
            self.fit(sentences)
        return sentences.apply(lambda x: self.vectorize_sentence(x, minlen, maxlen))

def get_vectorizer(vectorizer='tfidf'):
    if vectorizer == 'count':
        return CountVectorizer(tokenizer=tokenizing.tokenize,
                              preprocessor=preprocessing.preprocess)
    elif vectorizer == 'tfidf':
        return TfidfVectorizer(tokenizer=tokenizing.tokenize,
                              preprocessor=preprocessing.preprocess)
    elif vectorizer == 'word2vec':
        return Word2Vec(tokenizer=tokenizing.tokenize,
                        preprocessor=preprocessing.preprocess)