## Spark Application - execute with spark-submit

## Imports
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from sc_template import confugure_spark
from sc_vectorization import load_corpus, make_vectorizer


## Module Constants
APP_NAME = 'Text Classification'
CORPUS = '/home/python/project/data/hobbies/*/*.txt'


def make_classifier():
    stages = [
        # Create the vectorizer
        make_vectorizer(),
        # Index the labels of the classification
        StringIndexer(
            inputCol='label',
            outputCol='indexedLabel'
        ),
        # Create the classifier
        LogisticRegression(
            family='multinomial',
            maxIter=10,
            regParam=0.3,
            elasticNetParam=0.8,
            labelCol='indexedLabel',
            featuresCol='tfidf')
    ]
    return Pipeline(stages=stages)


def evaluate_classifier(classifier, predictions):
    # Show predictions
    (predictions
     .select('prediction', 'indexedLabel', 'tfidf')
     .show(5))

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol='indexedLabel',
        predictionCol='prediction',
        metricName='accuracy'
    )
    accuracy = evaluator.evaluate(predictions)
    print(f'Test Accuracy = {accuracy:.3f}')


## Main functionality
def main(sc, spark):
    # Load corpus
    corpus = load_corpus(sc, spark, CORPUS)

    # Split the data into train and test sets
    train, test = corpus.randomSplit([0.8, 0.2])

    # Fit classifier pipeline
    classifier = make_classifier()
    classifier = classifier.fit(train)

    # Make predictions
    predictions = classifier.transform(test)

    # Evaluate classifier
    evaluate_classifier(classifier, predictions)


if __name__ == '__main__':
    # Configure Spark
    sc, spark = confugure_spark(APP_NAME)

    # Execute Main functionality
    main(sc, spark)
