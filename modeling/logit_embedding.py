import config
import numpy as np
import pandas as pd
import feature_extraction
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Embedding, concatenate
from tensorflow.keras.callbacks import EarlyStopping

class LogitWithEmbedding:
    def __init__(self, embedding_dim=64, batch_size=32):
        self.vectorizer = feature_extraction.get('label_encoder')
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.classifier = self.build_classifier()

    def build_classifier(self):
        input_question = Input(shape = (config.MAX_LENGTH_QUESTION))
        input_text = Input(shape = (config.MAX_LENGTH_TEXT))
        
        with tf.name_scope("question_embedding"):
            question_embedding = Embedding(self.vectorizer.get_vocab_size(), self.embedding_dim)(input_question)
            question_flatten = Flatten()(question_embedding)
            
        with tf.name_scope("text_embedding"):
            text_embedding = Embedding(self.vectorizer.get_vocab_size(), self.embedding_dim)(input_text)
            text_flatten = Flatten()(text_embedding)
        
        merge = concatenate([question_flatten, text_flatten], axis = 1)
        
        dense_1 = Dense(10, activation = "relu")(merge)
        dense_2 = Dense(1, activation ="sigmoid")(dense_1)

        model = Model(inputs = [input_question, input_text], outputs = dense_2)
        model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["acc"])

        model.summary()
        return model
        
    def plot(self):
        return tf.keras.utils.plot_model(self.classifier, to_file='modeling/logit_embedding.png', show_shapes=True, show_layer_names=True)
        
    def DataGenerator(self, X, y):
        i = 0
        while True:
            if i == len(X):
                i = 0
                
            batch_size = min(self.batch_size, len(X) - i)
            X_question = X.iloc[i: i + batch_size]['question']
            X_text = X.iloc[i: i + batch_size]['text']
            y_batch = y.iloc[i: i + batch_size]
            self.vectorizer.fit(X_question)
            self.vectorizer.fit(X_text)
            X_question_embs = self.vectorizer.transform(X_question, maxlen=config.MAX_LENGTH_QUESTION)
            X_text_embs = self.vectorizer.transform(X_text, maxlen=config.MAX_LENGTH_TEXT)
            i += batch_size
            yield [np.array(X_question_embs), np.array(X_text_embs)], np.array(y_batch)

    def fit(self, X, y, epochs=50, early_stopping_rounds=7):
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)
        train_generator = self.DataGenerator(X_train, y_train)
        validation_generator = self.DataGenerator(X_val, y_val)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=early_stopping_rounds, monitor='val_acc', restore_best_weights=True)
            ]
        history = self.classifier.fit(train_generator, epochs=epochs, verbose=1, callbacks=callbacks,
            validation_data=validation_generator, steps_per_epoch=len(X_train)//self.batch_size,
            validation_steps=len(X_val)//self.batch_size
            )
        
        return history

    def predict(self, X):
        return self.predict_proba(X) >= 0.5

    def predict_proba(self, X):
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(np.reshape(X, (-1, 2)), columns=['question', 'text'])
            
        X_question_embs = self.vectorizer.transform(X['question'], maxlen=config.MAX_LENGTH_QUESTION)
        X_text_embs = self.vectorizer.transform(X['text'], maxlen=config.MAX_LENGTH_TEXT)
        y = self.classifier.predict([X_question_embs, X_text_embs])
        return y