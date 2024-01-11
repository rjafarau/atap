import nltk
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin


class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='english',
                 join=True, return_list=True):
        self.language = language
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.join = join
        self.return_list = return_list

    def is_punct(self, word):
        return all(
            unicodedata.category(char).startswith('P')
            for char in word
        )

    def is_stopword(self, word):
        return word.lower() in self.stopwords

    def normalize(self, document):
        return [
            self.lemmatize(word, tag).lower()
            for paragraph in document['content']
            for sentence in paragraph
            for word, tag in sentence
            if not self.is_punct(word)
            and not self.is_stopword(word)
        ]

    def lemmatize(self, token, pos_tag):
        tag = {
            'N': nltk.corpus.wordnet.NOUN,
            'V': nltk.corpus.wordnet.VERB,
            'R': nltk.corpus.wordnet.ADV,
            'J': nltk.corpus.wordnet.ADJ
        }.get(pos_tag[0], nltk.corpus.wordnet.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        normalized = map(self.normalize, documents)
        if self.join:
            normalized = map(' '.join, normalized)
        if self.return_list:
            normalized = list(normalized)
        return normalized
