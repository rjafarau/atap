## Spark Application - execute with spark-submit

## Imports
from sklearn.ensemble import AdaBoostClassifier

from sc_template import confugure_spark
from sc_vectorization import load_corpus, make_vectorizer


## Module Constants
APP_NAME = 'Scikit-Learn Sample Classifier'
CORPUS = '/home/python/project/data/hobbies/*/*.txt'


def make_accuracy_closure(model, correct, incorrect):
    # model should be a broadcast variable
    # correct and incorrect should be acculumators
    def inner(rows):
        X, y = [], []
        for row in rows:
            X.append(row['tfidf'])
            y.append(row['label'])

        yp = model.value.predict(X)
        for yi, ypi in zip(y, yp):
            if yi == ypi:
                correct.add(1)
            else:
                incorrect.add(1)
    return inner


## Main functionality
def main(sc, spark):
    # Load corpus
    corpus = load_corpus(sc, spark, CORPUS)

    # Fit vectorizer pipeline
    vectorizer = make_vectorizer()
    vectorizer = vectorizer.fit(corpus)

    # Create vectors
    vectors = vectorizer.transform(corpus)

    # Get the sample from the dataset
    sample = (
        vectors
        .sample(withReplacement=False,
                fraction=0.1,
                seed=42)
        .collect()
    )
    X = [row['tfidf'] for row in sample]
    y = [row['label'] for row in sample]

    # Train a Scikit-Learn Model
    clf = AdaBoostClassifier()
    clf.fit(X, y)

    # Broadcast the Scikit-Learn Model to the cluster
    clf = sc.broadcast(clf)

    # Create accumulators for correct vs incorrect
    correct = sc.accumulator(0)
    incorrect = sc.accumulator(1)

    # Create the accuracy closure
    accuracy = make_accuracy_closure(clf, incorrect, correct)

    # Compute the number incorrect and correct
    vectors.foreachPartition(accuracy)

    accuracy = correct.value / (correct.value + incorrect.value)
    print(f'Global accuracy of model was {accuracy:.3f}')


if __name__ == '__main__':
    # Configure Spark
    sc, spark = confugure_spark(APP_NAME)

    # Execute Main functionality
    main(sc, spark)
