from scipy import sparse

def concatenate_after_vectorizing(df, vectorizer, do_fit_vectorizer=True):
    if do_fit_vectorizer:
        for column in df.columns:
            vectorizer.fit(df[column])
    
    df_new = None
    for column in df.columns:
        df_new = sparse.hstack([df_new, vectorizer.transform(df[column])])
    return df_new