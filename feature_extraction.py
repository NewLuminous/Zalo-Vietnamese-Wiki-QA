import config
import numpy as np
import tokenization
import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

VECTORIZERS = ['bow', 'bow-ngram', 'tfidf', 'tfidf-ngram',
               'word2vec', 'word2vec-avg',
               'label_encoder']

class Word2Vec:
    def __init__(self, tokenizer, preprocessor):
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.model = None

    def fit(self, sentences):
        if self.model is None:
            #self.model = gensim.models.Word2Vec(sentences, min_count=10, size=config.WORD_VECTOR_DIM, workers=12, window=3, sg=1)
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
        return np.array([self.vectorize_sentence(sentence, minlen, maxlen) for sentence in sentences])
        
class Word2VecAvg(Word2Vec):
    def __init__(self, tokenizer, preprocessor):
        super().__init__(tokenizer, preprocessor)

    def vectorize_sentence(self, sentence):
        sentence = self.tokenizer(self.preprocessor(sentence))
        embs = []
        for token in sentence:
            if token in self.model.wv:
                embs.append(self.model.wv[token])
                
        if len(embs) == 0:
            embs.append(self.model.wv['&'])

        embs = np.array(embs).mean(axis=0)

        return embs
        
    def transform(self, sentences):
        if self.model is None:
            self.fit(sentences)
        return np.vstack([self.vectorize_sentence(sentence) for sentence in sentences])
        
# Handle input for tensorflow.keras.layers.Embedding layer
class LabelEncoder:
    def __init__(self, tokenizer, preprocessor):
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.model = Tokenizer(num_words=config.VOCAB_SIZE, oov_token="<OOV>", filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
        
    def get_word_index(self):
        return self.model.word_index
        
    def get_vocab_size(self):
        return self.model.num_words

    def fit(self, sentences):
        #sentences = [" ".join(self.tokenizer(self.preprocessor(sentence))) for sentence in sentences]
        #self.model.fit_on_texts(sentences)
        with open('saved_models/keras_tokenizer.pkl', 'rb') as f:
            self.model = pickle.load(f)
        return self

    def transform(self, sentences, minlen=None, maxlen=None):
        sentences = [" ".join(self.tokenizer(self.preprocessor(sentence))) for sentence in sentences]
        sequences = self.model.texts_to_sequences(sentences)
        padded_sequences = pad_sequences(sequences, maxlen=maxlen, truncating='post', padding='post')
        return padded_sequences

"""
    INPUT: n sentences of different lengths. The ith sentence has length m_i. v is the number of unique words in the corpus.
    OUTPUT:
    + CountVectorizer: A (n x v) matrix of token counts where each row represents a sentence.
        Eg: ['Hello World', 'Is that it? It is.']
        ---> [[1, 0, 0, 0, 1],
              [0, 2, 2, 1, 0]]
                                                        
    + TfidfVectorizer: A (n x v) matrix of TF-IDF features where each row represents a sentence.
        Eg: ['Hello World', 'Is that it? It is.']
        ---> [[0.70710678, 0.        , 0.        , 0.        , 0.70710678],
              [0.        , 0.66666667, 0.66666667, 0.33333333, 0.        ]]
                                                        
    + Word2Vec: n lists of k-dimensional vectors where each vector represents a word. k is determined by config.WORD_VECTOR_DIM.
        Eg: ['Hello World', 'Is that it? It is.']
        ---> [[[0.1364308 , -0.6438603 , 0.12640472 , -0.12537019],
               [0.03092231, -0.82507056, -0.18958199, -0.1303315 ]],
              
              [[0.30585364, -1.4214262 , -0.42150778, -0.00549458],
               [0.22600366, -1.6978164 , -0.750237  , -0.00871135],
               [0.33767852, -1.283297  , -0.37773284, -0.06668624],
               [0.33767852, -1.283297  , -0.37773284, -0.06668624],
               [0.30585364, -1.4214262 , -0.42150778, -0.00549458]]]
                                                
    + Word2VecAvg: A (n x k) matrix where each row represents a sentence. k is determined by config.WORD_VECTOR_DIM.
        Eg: ['Hello World', 'Is that it? It is.']
        ---> [[0.08367655, -0.7344654, -0.03158864, -0.12785085],
              [0.3026136 , -1.4214526, -0.46974367, -0.0306146 ]]
    
    + LabelEncoder: n sentences of same lengths due to padding. Each word is replaced by its index.
        Eg: ['Hello World', 'Is that it? It is.']
        ---> [[22964,  1844,     0,     0,     0],
              [ 5579,  9869,  6020,  6020,  5579]]
"""
def get(vectorizer='tfidf'):
    if vectorizer == 'bow':
        return CountVectorizer(tokenizer=tokenization.tokenize,
                              preprocessor=preprocessing.preprocess,
                              max_features=config.VOCAB_SIZE)
                              
    elif vectorizer == 'bow-ngram':
        return CountVectorizer(tokenizer=tokenization.tokenize,
                              preprocessor=preprocessing.preprocess,
                              max_features=config.VOCAB_SIZE,
                              ngram_range=(1, 2))
                              
    elif vectorizer == 'tfidf':
        return TfidfVectorizer(tokenizer=tokenization.tokenize,
                              preprocessor=preprocessing.preprocess,
                              max_features=config.VOCAB_SIZE)
                              
    elif vectorizer == 'tfidf-ngram':
        return TfidfVectorizer(tokenizer=tokenization.tokenize,
                              preprocessor=preprocessing.preprocess,
                              max_features=config.VOCAB_SIZE,
                              ngram_range=(1, 2))
                              
    elif vectorizer == 'word2vec':
        return Word2Vec(tokenizer=tokenization.tokenize,
                        preprocessor=preprocessing.preprocess)
                        
    elif vectorizer == 'word2vec-avg':
        return Word2VecAvg(tokenizer=tokenization.tokenize,
                           preprocessor=preprocessing.preprocess)
                           
    elif vectorizer == 'label_encoder':
        return LabelEncoder(tokenizer=tokenization.tokenize,
                            preprocessor=preprocessing.preprocess)
    else:
        raise Exception(f"Vectorizer '{vectorizer}' not found. Try 'feature_extraction.VECTORIZERS' for available vectorizers.")
        
if __name__ == "__main__":
    vectorizer = get('bow')
    vectorizer.fit(["It is true for all that that that that that that that refers to is not the same that that that that refers to."])
    print(vectorizer.transform(["That that is is that that is not is not. Is that it? It is."]))