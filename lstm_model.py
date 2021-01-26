import json
import random
from tqdm import tqdm

from pre_processor import PreProcessor

from collections import Counter

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Input

import numpy as np

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w,  t in zip(s['token'].values.tolist(), 
                                                           s['labeling'].values.tolist())]
        self.grouped = self.data.groupby('id').apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try: 
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None

class LSTMModel:
    
    def __init__(self, data, extraction_of):
        self.data = data
        self.extraction_of = extraction_of

    def fit_predict(self, testIds):
        print("START PREPROCESSING")
        vocabs = PreProcessor.getVocabs(self.data)
        
        train_data = dict()
        test_data = dict()

        for k, v in self.data.items():
            if k in testIds:
                test_data[k] = v
            else:
                train_data[k] = v

        train_data = PreProcessor.preprocess(train_data, self.extraction_of)
        test_data = PreProcessor.preprocess(test_data, self.extraction_of)

        words = list(set(vocabs))
        words.append("ENDPAD")

        n_words = len(words)

        tags = ['O', self.extraction_of]
        n_tags = len(tags)

        word2idx = {w: i for i, w in enumerate(words)}
        tag2idx = {t: i for i, t in enumerate(tags)}

        getter_train = SentenceGetter(train_data)
        sentences_train = getter_train.sentences

        getter_test = SentenceGetter(test_data)
        sentences_test = getter_test.sentences

        X_train = [[word2idx[w[0]] for w in s] for s in sentences_train]
        X_train = pad_sequences(maxlen=140, sequences=X_train, padding="post",value=n_words - 1)
        X_test = [[word2idx[w[0]] for w in s] for s in sentences_test]
        X_test = pad_sequences(maxlen=140, sequences=X_test, padding="post",value=n_words - 1)

        y_train = [[tag2idx[w[1]] for w in s] for s in sentences_train]
        y_train = pad_sequences(maxlen=140, sequences=y_train, padding="post", value=tag2idx["O"])
        y_train = [to_categorical(i, num_classes=n_tags) for i in y_train]

        y_test = [[tag2idx[w[1]] for w in s] for s in sentences_test]
        y_test = pad_sequences(maxlen=140, sequences=y_test, padding="post", value=tag2idx["O"])
        y_test = [to_categorical(i, num_classes=n_tags) for i in y_test]

        input = Input(shape=(140,))
        model = Embedding(input_dim=n_words, output_dim=50, input_length=140)(input)
        model = Dropout(0.1)(model)
        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
        out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)

        model = Model(input, out)

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=3, validation_split=0.2, verbose=1)

        y_true = []
        y_pred = []

        for i in tqdm(range(0, len(X_test))):
            p = model.predict(np.array(X_test[i]))
            p = np.argmax(p, axis=-1)

            t = y_test[i]

            for wordIdx, trueTagIdx, predTagIdx in zip(X_test[i], t, p[0]):
                tag_true = tags[np.argmax(trueTagIdx, axis=-1)]
                tag_pred = tags[predTagIdx]

                y_true.append(tag_true)
                y_pred.append(tag_pred)
        
        return y_true, y_pred