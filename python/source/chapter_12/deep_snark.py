import time
import functools

import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM
from keras.wrappers.scikit_learn import KerasClassifier

from reader import PickledCorpusReader
from transformer import (
    TextNormalizer, identity,
    KeyphraseExtractor, KeyphraseClipper
)

N_FEATURES = 5000
N_CLASSES = 4
DOC_LEN = 60


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Transforms sparse to dense"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.toarray()


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start
    return wrapper


def documents(corpus):
    return list(corpus.docs())


def continuous(corpus):
    return list(corpus.scores())


def make_categorical(corpus):
    """
    terrible : 0.0 < y <= 3.0
    okay     : 3.0 < y <= 5.0
    great    : 5.0 < y <= 7.0
    amazing  : 7.0 < y <= 10.1
    :param corpus:
    :return:
    """
    return np.digitize(continuous(corpus), [0.0, 3.0, 5.0, 7.0, 10.1])


def build_dnn():
    """
    Create a function that returns a compiled neural network
    :return: compiled Keras neural network model
    """
    nn = Sequential()
    nn.add(Dense(500, activation='relu', input_shape=(N_FEATURES,)))
    nn.add(Dense(150, activation='relu'))
    nn.add(Dense(N_CLASSES, activation='softmax'))
    nn.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return nn


def build_lstm():
    """
    Create a function that returns a compiled neural network
    :return: compiled Keras neural network model
    """
    nn = Sequential()
    nn.add(Embedding(N_FEATURES+1, 128, input_length=DOC_LEN))
    nn.add(Dropout(0.4))
    nn.add(LSTM(units=200, recurrent_dropout=0.2, dropout=0.2))
    nn.add(Dropout(0.2))
    nn.add(Dense(N_CLASSES, activation='softmax'))
    nn.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return nn


@timeit
def train_model(path, model, is_continuous=True, saveto=None, n_splits=10):
    """
    Trains model from corpus at specified path; constructing cross-validation
    scores using the n_splits parameter, then fitting the model on the full data and
    writing it to disk at the saveto path if specified. Returns the scores.
    """
    # Load the corpus data and labels for classification
    corpus = PickledCorpusReader(path)
    X = documents(corpus)
    if is_continuous:
        y = continuous(corpus)
        scoring = 'r2'
        cv = KFold(n_splits, shuffle=True, random_state=42)
    else:
        y = make_categorical(corpus)
        scoring = 'accuracy'
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=42)

    # Compute cross validation scores
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    # Fit the model on entire data set
    model.fit(X, y)

    # Write to disk if specified
    if saveto:
        joblib.dump(model, saveto)

    # Return scores as well as training time via decorator
    return scores


def fit_dnn_classifier(path, saveto=None):
    model = Pipeline([
        ('norm', TextNormalizer()),
        ('tfidf', TfidfVectorizer(tokenizer=identity,
                                  token_pattern=None,
                                  lowercase=False,
                                  max_features=N_FEATURES)), # need to control feature count
        ('dense', DenseTransformer()),
        ('dnn', KerasClassifier(build_fn=build_dnn, # pass but don't call the function!
                                epochs=200,
                                batch_size=128))
    ])

    if saveto is None:
        saveto = f'dnn_classifier_{time.time()}.pkl'

    scores, delta = train_model(path, model, is_continuous=False, saveto=saveto)
    for idx, score in enumerate(scores):
        print(f'Score on slice #{idx+1}: {score:0.3f}')
    print(f'CV score: {scores.mean():0.3f} ± {scores.std():0.3f}')
    print(f'Total fit time: {delta:0.3f} seconds')
    print(f'Model saved to {saveto}')


def fit_lstm_classifier(path, saveto=None):
    model = Pipeline([
        ('extractor', KeyphraseExtractor(return_list=True)),
        ('clipper', KeyphraseClipper(n_features=N_FEATURES,
                                     doc_len=DOC_LEN)),
        ('lstm', KerasClassifier(build_fn=build_lstm,
                                 epochs=20,
                                 batch_size=128))
    ])

    if saveto is None:
        saveto = f'lstm_classifier_{time.time()}.pkl'

    scores, delta = train_model(path, model, is_continuous=False, saveto=saveto)
    for idx, score in enumerate(scores):
        print(f'Score on slice #{idx+1}: {score:0.3f}')
    print(f'CV score: {scores.mean():0.3f} ± {scores.std():0.3f}')
    print(f'Total fit time: {delta:0.3f} seconds')
    print(f'Model saved to {saveto}')
