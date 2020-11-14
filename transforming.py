import config
import vectorizing
import numpy as np
from scipy import sparse

def vectorize_and_concatenate(df, vectorizer, do_fit_vectorizer=True):
    if do_fit_vectorizer:
        for column in df.columns:
            vectorizer.fit(df[column])
    
    df_new = None
    for column in df.columns:
        df_new = sparse.hstack([df_new, vectorizer.transform(df[column])])
    return df_new
    
def vectorize_and_concatenate_qa(df, vectorizer, do_fit_vectorizer=True):
    # Size of vector depends on the input sentence's length
    if type(vectorizer) in [vectorizing.Word2Vec, vectorizing.LabelEncoder]:
        if do_fit_vectorizer:
            for column in df.columns:
                vectorizer.fit(df[column])
            
        df_question_embs = vectorizer.transform(df['question'],
                                                minlen=config.MAX_LENGTH_QUESTION,
                                                maxlen=config.MAX_LENGTH_QUESTION)
        df_text_embs = vectorizer.transform(df['text'],
                                            minlen=config.MAX_LENGTH_TEXT,
                                            maxlen=config.MAX_LENGTH_TEXT)
                                            
        if type(vectorizer) is vectorizing.Word2Vec:
            df_question_embs = np.vstack([row.flatten() for row in df_question_embs])
            df_text_embs = np.vstack([row.flatten() for row in df_text_embs])
            
        return np.hstack([df_question_embs, df_text_embs])
    else:
        return vectorize_and_concatenate(df, vectorizer, do_fit_vectorizer)