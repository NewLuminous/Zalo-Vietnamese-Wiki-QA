import config
import os
import numpy as np
import pandas as pd
import vectorizing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, Bidirectional, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

class CRNN:
    def __init__(self, question_encoder_shape=20, text_encoder_shape=70, learning_rate=0.001):
        self.vectorizer = vectorizing.get_vectorizer('word2vec')
        self.num_features = config.WORD_VECTOR_DIM
        self.question_encoder_shape = question_encoder_shape
        self.text_encoder_shape = text_encoder_shape
        # Tensorflow currently does not support Tensors with different lengths along a dimension.
        self.batch_size = 1
        self.learning_rate = learning_rate

        self.model = self.build_model()

    def build_model(self):
        input_question = Input(shape = (None, self.num_features))
        input_ans = Input(shape = (None, self.num_features))

        conv1d_1 = Conv1D(200, 5, activation = "tanh", kernel_regularizer = l2(0.0001))(input_question)
        max_pooling_1 = MaxPooling1D()(conv1d_1)
        question_encoder = Bidirectional(LSTM(self.question_encoder_shape, ))(max_pooling_1)
        
        conv1d_2 = Conv1D(400, 5, activation = "tanh", kernel_regularizer = l2(0.0001))(input_ans)
        max_pooling_2 = MaxPooling1D()(conv1d_2)
        conv1d_3 = Conv1D(600, 5, activation = "tanh", kernel_regularizer = l2(0.0001))(max_pooling_2)
        max_pooling_3 = MaxPooling1D()(conv1d_3)
        text_encoder = Bidirectional(LSTM(self.text_encoder_shape))(max_pooling_3)
        
        merge = concatenate([question_encoder, text_encoder])
        
        dense_1 = Dense(128, activation = "relu")(merge)
        dropout_1 = Dropout(0.25)(dense_1)
        dense_2 = Dense(256, activation = "relu")(dropout_1)
        dropout_2 = Dropout(0.25)(dense_2)
        dense_3 = Dense(1, activation ="sigmoid")(dropout_2)

        model = Model(inputs = [input_question, input_ans], outputs = dense_3)
        model.compile(optimizer = Adam(self.learning_rate), loss = "binary_crossentropy", metrics = ["acc"])

        model.summary()
        return model
        
    def plot(self):
        return tf.keras.utils.plot_model(self.model, to_file='modeling/crnn.png', show_shapes=True, show_layer_names=True)
        
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
            X_question_embs = self.vectorizer.transform(X_question, minlen=config.MIN_LENGTH_QUESTION)
            X_text_embs = self.vectorizer.transform(X_text, minlen=config.MIN_LENGTH_TEXT)
            if batch_size == 1:
                X_question_embs = np.expand_dims(np.array(X_question_embs)[0], axis=0)
                X_text_embs = np.expand_dims(np.array(X_text_embs)[0], axis=0)
            i += batch_size
            yield [np.array(X_question_embs), np.array(X_text_embs)], np.array(y_batch)

    def fit(self, X, y, epochs=50, early_stopping_rounds=7):
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)
        train_generator = self.DataGenerator(X_train, y_train)
        validation_generator = self.DataGenerator(X_val, y_val)

        model_dir = './checkpoints/'
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=early_stopping_rounds, monitor='val_acc', restore_best_weights=True),
            tf.keras.callbacks.TensorBoard(log_dir = "./logs"),
            tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(model_dir, "weights-epoch{epoch:02d}-loss{val_loss:.2f}-acc{val_acc:.2f}.h5"))
            ]
        history = self.model.fit(train_generator, epochs=epochs, verbose=1, callbacks=callbacks,
            validation_data=validation_generator, steps_per_epoch=len(X_train)//self.batch_size,
            validation_steps=len(X_val)//self.batch_size
            )
        
        return history

    def predict(self, X):
        return self.predict_proba(X) >= 0.5

    def predict_proba(self, X):
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(np.reshape(X, (-1, 2)), columns=['question', 'text'])
            
        X_question_embs = self.vectorizer.transform(X['question'], minlen=config.MIN_LENGTH_QUESTION)
        X_text_embs = self.vectorizer.transform(X['text'], minlen=config.MIN_LENGTH_TEXT)
        X_question_embs = X_question_embs.apply(lambda row: np.expand_dims(np.array(row), axis=0))
        X_text_embs = X_text_embs.apply(lambda row: np.expand_dims(np.array(row), axis=0))
        X_embs = pd.concat([X_question_embs, X_text_embs], axis=1)
        y = X_embs.apply(lambda row: self.model.predict([row['question'], row['text']])[0], axis=1)
        return y