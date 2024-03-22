## Spark Application - execute with spark-submit

## Imports
import os

from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, NGram
from pyspark.ml.feature import (
    Tokenizer, RegexTokenizer,
    HashingTF, IDF
)

from sc_template import confugure_spark


## Module Constants
APP_NAME = 'Text Vectorization'
CORPUS = '/home/python/project/data/hobbies/*/*.txt'


## Closure Functions
def parse_label(path):
    # Returns the name of the directory containing the file
    return os.path.basename(os.path.dirname(path))


## Data Manipulation
def load_corpus(sc, spark, path):
    # Load data from disk and transform into a DataFrame
    data = sc.wholeTextFiles(path)
    corpus = data.map(lambda d: (parse_label(d[0]), d[1]))
    return spark.createDataFrame(corpus, ['label', 'text'])


def make_vectorizer(stopwords=True, tfidf=True, n_features=4096):
    # Creates a vectorization pipeline that starts with tokenization
    stages = [
        Tokenizer(
            inputCol='text',
            outputCol='tokens'
        )
    ]

    # Append stopwords to the pipeline if requested
    if stopwords:
        stages.append(
            StopWordsRemover(
                caseSensitive=False,
                inputCol=stages[-1].getOutputCol(),
                outputCol='filtered_tokens',
            )
        )

    # Create the Hashing term frequency vectorizer
    stages.append(
        HashingTF(
            numFeatures=n_features,
            inputCol=stages[-1].getOutputCol(),
            outputCol='frequency'
        )
    )

    # Append the IDF vectorizer if requested
    if tfidf:
        stages.append(
            IDF(
                inputCol=stages[-1].getOutputCol(),
                outputCol='tfidf'
            )
        )

    # Return the completed pipeline
    return Pipeline(stages=stages)


## Main functionality
def main(sc, spark):
    # Load corpus
    corpus = load_corpus(sc, spark, CORPUS)
    
    # Fit vectorizer pipeline
    vectorizer = make_vectorizer()
    vectorizer = vectorizer.fit(corpus)

    # Create vectors
    vectors = vectorizer.transform(corpus)
    print(vectors.head())


if __name__ == '__main__':
    # Configure Spark
    sc, spark = confugure_spark(APP_NAME)

    # Execute Main functionality
    main(sc, spark)
