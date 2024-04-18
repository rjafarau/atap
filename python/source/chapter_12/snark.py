import time
import logging
import functools

import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

from reader import PickledCorpusReader
from transformer import TextNormalizer, identity


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(processName)-10s %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


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


@timeit
def train_model(path, model, is_continuous=True, saveto=None, cv=10):
    """
    Trains model from corpus at specified path; constructing cross-validation
    scores using the cv parameter, then fitting the model on the full data and
    writing it to disk at the saveto path if specified. Returns the scores.
    """
    # Load the corpus data and labels for classification
    corpus = PickledCorpusReader(path)
    X = documents(corpus)
    if is_continuous:
        y = continuous(corpus)
        scoring = 'r2'
    else:
        y = make_categorical(corpus)
        scoring = 'f1_weighted'

    # Compute cross validation scores
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    # Fit the model on entire data set
    model.fit(X, y)

    # Write to disk if specified
    if saveto:
        joblib.dump(model, saveto)

    # Return scores as well as training time via decorator
    return scores


def fit_mlp_classifier(path, saveto=None):
    model = Pipeline([
        ('norm', TextNormalizer()),
        ('tfidf', TfidfVectorizer(tokenizer=identity,
                                  token_pattern=None,
                                  lowercase=False)),
        ('ann', MLPClassifier(hidden_layer_sizes=(500, 150)))
    ])

    if saveto is None:
        saveto = 'mlp_classifier_{}.pkl'.format(time.time())

    scores, delta = train_model(path, model, is_continuous=False, saveto=saveto)
    logging.info((
        f'mlp classifier training took {delta:0.2f} seconds '
        f'with an average score of {scores.mean():0.3f}'
    ))


def fit_mlp_regressor(path, saveto=None):
    model = Pipeline([
        ('norm', TextNormalizer()),
        ('tfidf', TfidfVectorizer(tokenizer=identity,
                                  token_pattern=None,
                                  lowercase=False)),
        ('ann', MLPRegressor(hidden_layer_sizes=(500, 150)))
    ])

    if saveto is None:
        saveto = 'mlp_regressor_{}.pkl'.format(time.time())

    scores, delta = train_model(path, model, is_continuous=True, saveto=saveto)
    logging.info((
        f'mlp regressor training took {delta:0.2f} seconds '
        f'with an average score of {scores.mean():0.3f}'
    ))
