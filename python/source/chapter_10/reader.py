import os
import time
import pickle
import logging

import bs4
import nltk

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

from readability.readability import Unparseable
from readability.readability import Document as Paper

logger = logging.getLogger("readability.readability")
logger.disabled = True

DOC_PATTERN = r'(?!\.)[a-z_\s]+/[\w\s\d\-]+\.html'
PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f\d\-]+\.pickle'
CAT_PATTERN = r'([a-z_\s]+)/.*'

# Tags to extract as paragraphs from the HTML text
TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']


class HTMLCorpusReader(CategorizedCorpusReader, CorpusReader):
    """
    A corpus reader for raw HTML documents to enable preprocessing.
    """

    def __init__(self, root, fileids=DOC_PATTERN,
                 word_tokenizer=nltk.WordPunctTokenizer(),
                 sent_tokenizer=nltk.data.load(
                     'tokenizers/punkt/english.pickle'
                 ),
                 pos_tagger=nltk.PerceptronTagger(),
                 tags=TAGS, encoding='latin-1', **kwargs):
        """
        Initialize the corpus reader. Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor. The remaining
        arguments are passed to the ``CorpusReader`` constructor.
        """
        # Add the default category pattern if not passed into the class.
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

        # Initialize the NLTK corpus reader objects
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids, encoding)

        self._word_tokenizer = word_tokenizer
        self._sent_tokenizer = sent_tokenizer
        self._pos_tagger = pos_tagger
        self._tags = tags

    def resolve(self, fileids, categories):
        """
        Returns a list of fileids or categories depending on what is passed
        to each internal corpus reader function. Implemented similarly to
        the NLTK ``CategorizedPlaintextCorpusReader``.
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Returns the complete text of an HTML document, closing the document
        after we are done reading it and yielding it in a memory safe fashion.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with open(path, 'r', encoding=encoding) as f:
                yield f.read()

    def html(self, fileids=None, categories=None):
        """
        Returns the HTML content of each document, cleaning it using
        the readability-lxml library.
        """
        for doc in self.docs(fileids, categories):
            try:
                yield Paper(doc).summary()
            except Unparseable as e:
                print("Could not parse HTML: {}".format(e))
                continue

    def paras(self, fileids=None, categories=None):
        """
        Uses BeautifulSoup to parse the paragraphs from the HTML.
        """
        for html in self.html(fileids, categories):
            soup = bs4.BeautifulSoup(html, 'lxml')
            for element in soup.find_all(self._tags):
                yield element.text
            soup.decompose()

    def titles(self, fileids=None, categories=None):
        """
        Uses BeautifulSoup to identify titles from the
        head tags within the HTML
        """
        for doc in self.docs(fileids, categories):
            soup = bs4.BeautifulSoup(doc, 'lxml')
            try:
                yield soup.title.text
                soup.decompose()
            except AttributeError as e:
                continue

    def sents(self, fileids=None, categories=None):
        """
        Uses the built in sentence tokenizer to extract sentences from the
        paragraphs. Note that this method uses BeautifulSoup to parse HTML.
        """
        for paragraph in self.paras(fileids, categories):
            for sentence in self._sent_tokenizer.tokenize(paragraph):
                yield sentence

    def words(self, fileids=None, categories=None):
        """
        Uses the built in word tokenizer to extract words from sentences.
        Note that this method uses BeautifulSoup to parse HTML content.
        """
        for sentence in self.sents(fileids, categories):
            for word in self._word_tokenizer.tokenize(sentence):
                yield word

    def tokenize(self, fileids=None, categories=None):
        """
        Segments, tokenizes, and tags a document in the corpus.
        """
        for paragraph in self.paras(fileids, categories):
            yield [
                self._pos_tagger.tag(self._word_tokenizer.tokenize(sentence))
                for sentence in self._sent_tokenizer.tokenize(paragraph)
            ]

    def sizes(self, fileids=None, categories=None):
        """
        Returns a list of tuples, the fileid and size on disk of the file.
        This function is used to detect oddly large files in the corpus.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, getting every path and computing filesize
        for path in self.abspaths(fileids):
            yield os.path.getsize(path)

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
        for para in self.paras(fileids, categories):
            counts['paras'] += 1

            for sent in self._sent_tokenizer.tokenize(para):
                counts['sents'] += 1

                for word in self._word_tokenizer.tokenize(sent):
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
