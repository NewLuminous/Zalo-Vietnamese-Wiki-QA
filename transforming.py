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
    if type(vectorizer) is vectorizing.Word2Vec:
        if do_fit_vectorizer:
            for column in df.columns:
                vectorizer.fit(df[column])
            
        df_question_embs = vectorizer.transform(df['question'], minlen=30, maxlen=30)
        df_answer_embs = vectorizer.transform(df['answer'], minlen=500, maxlen=500)
        df_question_flat = np.vstack(df_question_embs.apply(lambda row: row.flatten()))
        df_answer_flat = np.vstack(df_answer_embs.apply(lambda row: row.flatten()))
        return np.hstack([df_question_flat, df_answer_flat])
    else:
        return vectorize_and_concatenate(df, vectorizer, do_fit_vectorizer)