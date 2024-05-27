import itertools
import unicodedata

import nltk

from keras.utils.data_utils import pad_sequences
from sklearn.base import BaseEstimator, TransformerMixin

GRAMMAR = r'KT: {(<RB.> <JJ.*>|<VB.*>|<RB.*>)|(<JJ> <NN.*>)}'


class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='english',
                 join=False, return_list=False):
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
            for paragraph in document['text']
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
        result = map(self.normalize, documents)
        if self.join:
            result = map(' '.join, result)
        if self.return_list:
            result = list(result)
        return result


def identity(doc):
    return doc


class KeyphraseExtractor(BaseEstimator, TransformerMixin):
    """
    Extract adverbial and adjective phrases, and transform
    documents into lists of these keyphrases, with a total
    keyphrase lexicon limited by the nfeatures parameter
    and a document length limited/padded to doclen
    """
    def __init__(self, grammar=GRAMMAR, return_list=False):
        self.grammar = grammar
        self.return_list = return_list
        self.chunker = nltk.RegexpParser(self.grammar)

    def normalize(self, sentence):
        """
        Removes punctuation from a tokenized/tagged sentence and
        lowercases words.
        """
        return [(word.lower(), tag)
                for word, tag in sentence
                if not all(unicodedata.category(char).startswith('P')
                           for char in word)]

    def extract_keyphrases(self, document):
        """
        For a document, parse sentences using our chunker created by
        our grammar, converting the parse tree into a tagged sequence.
        Extract phrases, rejoin with a space, and yield the document
        represented as a list of it's keyphrases.
        """
        keyphrases = []
        for paragraph in document['text']:
            for sentence in paragraph:
                sentence = self.normalize(sentence)
                if not sentence:
                    continue
                chunks = nltk.tree2conlltags(
                    self.chunker.parse(sentence)
                )
                keyphrases.extend([
                    ' '.join(word for word, pos, chunk in group)
                    for key, group in itertools.groupby(
                        chunks, lambda term: term[-1] != 'O'
                    ) if key
                ])
        return keyphrases

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        result = map(self.extract_keyphrases, documents)
        if self.return_list:
            result = list(result)
        return result


class KeyphraseClipper(BaseEstimator, TransformerMixin):

    def __init__(self, n_features=100000, doc_len=60):
        self.n_features = n_features
        self.doc_len = doc_len

    def get_lexicon(self, documents):
        """
        Build a lexicon of size n_features
        """
        fdist = nltk.FreqDist(
            keyphrase
            for doc in documents
            for keyphrase in doc
        )
        most_common = fdist.most_common(self.n_features)
        return {keyphrase: i
                for i, (keyphrase, _) in enumerate(most_common, 1)}

    def fit(self, documents, labels=None):
        self.lexicon = self.get_lexicon(documents)
        return self

    def clip(self, document):
        """
        Remove keyphrases from documents that aren't in the lexicon
        """
        return [self.lexicon[keyphrase]
                for keyphrase in document
                if keyphrase in self.lexicon]

    def transform(self, documents):
        return pad_sequences(
            sequences=list(map(self.clip, documents)),
            maxlen=self.doc_len
        )
