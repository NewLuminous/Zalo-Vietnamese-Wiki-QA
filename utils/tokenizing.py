from pyvi import ViTokenizer

def tokenize(text):
    return ViTokenizer.tokenize(text).split()