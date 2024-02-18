import time
import pickle

import nltk

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f\d\-]+\.pickle'
CAT_PATTERN = r'([a-z_\s]+)/.*'


class HTMLPickledCorpusReader(CategorizedCorpusReader, CorpusReader):

    def __init__(self, root, fileids=PKL_PATTERN, **kwargs):
        """
        Initialize the corpus reader. Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor. The remaining arguments
        are passed to the ``CorpusReader`` constructor.
        """
        # Add the default category pattern if not passed into the class.
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids)

    def resolve(self, fileids, categories):
        """
        Returns a list of fileids or categories depending on what is passed
        to each internal corpus reader function. This primarily bubbles up to
        the high level ``docs`` method, but is implemented here similar to
        the nltk ``CategorizedPlaintextCorpusReader``.
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Returns the document loaded from a pickled object for every file in
        the corpus. Similar to the BaleenCorpusReader, this uses a generator
        to achive memory safe iteration.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path in self.abspaths(fileids):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def titles(self, fileids=None, categories=None):
        """
        Uses BeautifulSoup to identify titles from the
        head tags within the HTML
        """
        for doc in self.docs(fileids, categories):
            yield doc['title']

    def tagged_paras(self, fileids=None, categories=None):
        """
        Returns a generator of paragraphs where each paragraph is a list of
        sentences, which is in turn a list of (word, tag) tuples.
        """
        for doc in self.docs(fileids, categories):
            for tagged_para in doc['content']:
                yield tagged_para

    def paras(self, fileids=None, categories=None):
        """
        Returns a generator of paragraphs where each paragraph is a list of
        sentences, which is in turn a list of words.
        """
        for tagged_para in self.tagged_paras(fileids, categories):
            yield [[word for word, tag in tagged_sent]
                   for tagged_sent in tagged_para]

    def tagged_sents(self, fileids=None, categories=None):
        """
        Returns a generator of sentences where each sentence is a list of
        (word, tag) tuples.
        """
        for tagged_para in self.tagged_paras(fileids, categories):
            for tagged_sent in tagged_para:
                yield tagged_sent

    def sents(self, fileids=None, categories=None):
        """
        Returns a generator of sentences where each sentence is a list of
        words.
        """
        for tagged_sent in self.tagged_sents(fileids, categories):
            yield [word for word, tag in tagged_sent]

    def tagged_words(self, fileids=None, categories=None):
        """
        Returns a generator of (word, tag) tuples.
        """
        for sent in self.tagged_sents(fileids, categories):
            for word, tag in sent:
                yield word, tag

    def words(self, fileids=None, categories=None):
        """
        Returns a generator of words.
        """
        for word, tag in self.tagged_words(fileids, categories):
            yield word

    def describe(self, fileids=None, categories=None):
        """
        Performs a single pass of the corpus and returns a dictionary with a
        variety of metrics concerning the state of the corpus.
        """
        started = time.perf_counter()

        # Structures to perform counting.
        counts = nltk.FreqDist()
        words = nltk.FreqDist()

        # Perform single pass over paragraphs, tokenize and count
        for para in self.tagged_paras(fileids, categories):
            counts['paras'] += 1

            for sent in para:
                counts['sents'] += 1

                for word, tag in sent:
                    counts['words'] += 1
                    words[word] += 1

        # Compute the number of files and categories in the corpus
        n_fileids = len(self.resolve(fileids, categories) or self.fileids())
        n_categories = len(self.categories(self.resolve(fileids, categories)))

        # Return data structure with information
        return {
            'files': n_fileids,
            'categories': n_categories,
            'paras': counts['paras'],
            'sents': counts['sents'],
            'words': counts['words'],
            'vocab': len(words),
            'lexdiv': counts['words'] / len(words),
            'ppdoc': counts['paras'] / n_fileids,
            'sppar': counts['sents'] / counts['paras'],
            'secs': time.perf_counter() - started,
        }

    def describes(self, fileids=None, categories=None):
        """
        Returns a string representation of the describe command.
        """
        return (
            "HTML corpus contains {files:,} files in {categories:,} categories.\n"
            "Structured as:\n"
            "    {paras:,} paragraphs ({ppdoc:0,.3f} mean paragraphs per file)\n"
            "    {sents:,} sentences ({sppar:0,.3f} mean sentences per paragraph).\n"
            "Word count of {words:,} with a vocabulary of {vocab:,} "
            "({lexdiv:0,.3f} lexical diversity).\n"
            "Corpus scan took {secs:0,.3f} seconds."
        ).format(**self.describe(fileids, categories))
