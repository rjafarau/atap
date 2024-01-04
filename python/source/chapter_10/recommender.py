import joblib

from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from transformer import TextNormalizer


class KNNTransformer(NearestNeighbors, TransformerMixin):
    """
    Scikit-Learn's KNN doesn't have a transform method,
    so give it one.
    """
    def transform(self, documents):
        return self.kneighbors(documents, return_distance=False)


class KNNRecommender(BaseEstimator, TransformerMixin):
    """
    Given input terms, provide k recipe recommendations
    """
    def __init__(self,
                 n_components=100,
                 n_neighbors=3,
                 algorithm='auto'):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.pipeline = Pipeline([
            ('normalizer', TextNormalizer()),
            ('vectorizer', TfidfVectorizer()),
            ('svd', TruncatedSVD(n_components=self.n_components)),
            ('knn', KNNTransformer(n_neighbors=self.n_neighbors,
                                   algorithm=self.algorithm))
        ])

    def load(self, path):
        """
        Load a pickled knn recommender from disk, if it exists
        """
        self.pipeline = joblib.load(path)
        return self

    def save(self, path):
        """
        It takes a long time to fit, so just do it once!
        """
        joblib.dump(self.pipeline, path)

    def fit(self, documents):
        self.pipeline.fit(documents)
        return self

    def transform(self, documents):
        return self.pipeline.transform(documents)
