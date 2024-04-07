import time
# import pickle
import sqlite3

import nltk

# from nltk.corpus.reader.api import CorpusReader

# PKL_PATTERN = r'(?!\.)[\w\s\d\-]+\.pickle'


class SqliteCorpusReader(object):

    def __init__(self, path,
                 word_tokenizer=nltk.WordPunctTokenizer(),
                 sent_tokenizer=nltk.data.load(
                     'tokenizers/punkt/english.pickle'
                 ),
                 pos_tagger=nltk.PerceptronTagger()):
        self._cur = sqlite3.connect(path).cursor()
        self._word_tokenizer = word_tokenizer
        self._sent_tokenizer = sent_tokenizer
        self._pos_tagger = pos_tagger

    def get_rows(self, query):
        for row in self._cur.execute(query):
            yield row

    def scores(self):
        """
        Returns the review score
        """
        return self.get_rows('SELECT score FROM reviews')

    def texts(self):
        """
        Returns the full review texts
        """
        return self.get_rows('SELECT content FROM content')

    def ids(self):
        """
        Returns the review ids
        """
        return self.get_rows('SELECT reviewid FROM content')

    def ids_and_texts(self):
        """
        Returns the review ids and texts
        """
        return self.get_rows('SELECT * FROM content')

    def scores_albums_artists_texts(self):
        """
        Returns a generator with each review represented as a
        (score, album name, artist name, review text) tuple
        """
        return self.get_rows("""
            SELECT R.score, L.label, A.artist, C.content
            FROM reviews R
            JOIN labels L ON R.reviewid = L.reviewid
            JOIN artists A ON L.reviewid = A.reviewid
            JOIN content C ON A.reviewid = C.reviewid
        """)

    def albums(self):
        """
        Returns the names of albums being reviewed
        """
        return self.get_rows('SELECT * FROM labels')

    def artists(self):
        """
        Returns the name of the artist being reviewed
        """
        return self.get_rows('SELECT * FROM artists')

    def genres(self):
        """
        Returns the music genre of each review
        """
        return self.get_rows('SELECT * FROM genres')

    def years(self):
        """
        Returns the publication year of each review

        Note: There are many missing values
        """
        return self.get_rows('SELECT * FROM years')

    def paras(self):
        """
        Returns a generator of paragraphs.
        """
        for text in self.texts():
            for paragraph in text:
                yield paragraph

    def sents(self):
        """
        Returns a generator of sentences.
        """
        for paragraph in self.paras():
            for sentence in self._sent_tokenizer.tokenize(paragraph):
                yield sentence

    def words(self):
        """
        Returns a generator of words.
        """
        for sentence in self.sents():
            for word in self._word_tokenizer.tokenize(sentence):
                yield word

    def tokenize(self, text):
        """
        Segments, tokenizes, and tags a document in the corpus. Returns a
        generator of paragraphs, which are lists of sentences, which in turn
        are lists of part of speech tagged words.
        """
        for paragraph in text:
            yield [
                self._pos_tagger.tag(self._word_tokenizer.tokenize(sentence))
                for sentence in self._sent_tokenizer.tokenize(paragraph)
            ]

    def describe(self):
        """
        Performs a single pass of the corpus and returns a dictionary with a
        variety of metrics concerning the state of the corpus.
        """
        started = time.perf_counter()

        # Structures to perform counting.
        counts = nltk.FreqDist()
        words = nltk.FreqDist()

        # Perform single pass over paragraphs, tokenize and count
        for para in self.paras():
            counts['paras'] += 1

            for sent in self._sent_tokenizer.tokenize(para):
                counts['sents'] += 1

                for word in self._word_tokenizer.tokenize(sent):
                    counts['words'] += 1
                    words[word] += 1

        # Compute the number of files
        n_fileids = sum(1 for _ in self.ids())

        # Return data structure with information
        return {
            'files': n_fileids,
            'paras': counts['paras'],
            'sents': counts['sents'],
            'words': counts['words'],
            'vocab': len(words),
            'lexdiv': counts['words'] / len(words),
            'ppdoc': counts['paras'] / n_fileids,
            'sppar': counts['sents'] / counts['paras'],
            'secs': time.perf_counter() - started,
        }

    def describes(self):
        """
        Returns a string representation of the describe command.
        """
        return (
            "HTML corpus contains {files:,} files.\n"
            "Structured as:\n"
            "    {paras:,} paragraphs ({ppdoc:0,.3f} mean paragraphs per file)\n"
            "    {sents:,} sentences ({sppar:0,.3f} mean sentences per paragraph).\n"
            "Word count of {words:,} with a vocabulary of {vocab:,} "
            "({lexdiv:0,.3f} lexical diversity).\n"
            "Corpus scan took {secs:0,.3f} seconds."
        ).format(**self.describe())


if __name__ == '__main__':
    corpus = SqliteCorpusReader('/home/python/project/data/database.sqlite')
    print(corpus.describes())

# class PickledReviewsReader(CorpusReader):
#     def __init__(self, root, fileids=PKL_PATTERN, **kwargs):
#         """
#         Initialize the corpus reader
#         """
#         CorpusReader.__init__(self, root, fileids, **kwargs)

#     def texts_scores(self, fileids=None):
#         """
#         Returns the document loaded from a pickled object for every file in
#         the corpus. Similar to the SqliteCorpusReader, this uses a generator
#         to achieve memory safe iteration.
#         """
#         # Create a generator, loading one document into memory at a time.
#         for path, enc, fileid in self.abspaths(fileids, True, True):
#             with open(path, 'rb') as f:
#                 yield pickle.load(f)

#     def reviews(self, fileids=None):
#         """
#         Returns a generator of paragraphs where each paragraph is a list of
#         sentences, which is in turn a list of (token, tag) tuples.
#         """
#         for text,score in self.texts_scores(fileids):
#             yield text

#     def scores(self, fileids=None):
#         """
#         Return the scores
#         """
#         for text,score in self.texts_scores(fileids):
#             yield score

#     def paras(self, fileids=None):
#         """
#         Returns a generator of paragraphs where each paragraph is a list of
#         sentences, which is in turn a list of (token, tag) tuples.
#         """
#         for review in self.reviews(fileids):
#             for paragraph in review:
#                 yield paragraph

#     def sents(self, fileids=None):
#         """
#         Returns a generator of sentences where each sentence is a list of
#         (token, tag) tuples.
#         """
#         for paragraph in self.paras(fileids):
#             for sentence in paragraph:
#                 yield sentence

#     def tagged(self, fileids=None):
#         for sent in self.sents(fileids):
#             for token in sent:
#                 yield token

#     def words(self, fileids=None):
#         """
#         Returns a generator of (token, tag) tuples.
#         """
#         for token in self.tagged(fileids):
#             yield token[0]


# if __name__ == '__main__':
#     # Download the data from https://www.kaggle.com/nolanbconaway/pitchfork-data/data
#     # preprocess by running preprocess.py to produce pickled corpus
#     reader = PickledReviewsReader('../review_corpus_proc')
#     print(len(list(reader.reviews())))