## Spark Application - execute with spark-submit

## Imports
import tabulate

from pyspark.ml import Pipeline
from pyspark.ml.feature import Word2Vec, Tokenizer
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from sc_template import confugure_spark
from sc_vectorization import load_corpus


## Module Constants
APP_NAME = 'Text Clustering'
CORPUS = '/home/python/project/data/hobbies/*/*.txt'


def make_clusterer():
    # Creates the vector/cluster pipeline
    stages=[
        Tokenizer(
            inputCol='text',
            outputCol='tokens'
        ),
        Word2Vec(
            vectorSize=7,
            minCount=0,
            inputCol='tokens',
            outputCol='vecs'
        ),
        BisectingKMeans(
            k=10,
            maxIter=10,
            seed=42,
            featuresCol='vecs',
            predictionCol='prediction'
        )
    ]
    return Pipeline(stages=stages)


def evaluate_clusterer(clusterer, predictions):
    # Evaluate clusterer
    bkm = clusterer.stages[-1]
    cost = bkm.summary.trainingCost
    sizes = bkm.summary.clusterSizes

    evaluator = ClusteringEvaluator(
        featuresCol='vecs',
        predictionCol='prediction'
    )
    silhouette = evaluator.evaluate(predictions)

    # Get the text representation of each cluster
    wvec = clusterer.stages[-2]
    table = [['Cluster', 'Size', 'Terms']]
    for ci, c in enumerate(bkm.clusterCenters()):
        ct = wvec.findSynonyms(c, 7)
        size = sizes[ci]
        terms = ' '.join([row.word for row in ct.take(7)])
        table.append([ci, size, terms])

    # Print results
    print(tabulate.tabulate(table, tablefmt='simple', headers='firstrow'))
    print(f'Sum of square distance to center: {cost:.3f}')
    print(f'Silhouette with squared euclidean distance: {silhouette:.3f}')


## Main functionality
def main(sc, spark):
    # Load corpus
    corpus = load_corpus(sc, spark, CORPUS)

    # Fit clusterer pipeline
    clusterer = make_clusterer()
    clusterer = clusterer.fit(corpus)

    # Make predictions
    predictions = clusterer.transform(corpus)

    # Evaluate clusterer
    evaluate_clusterer(clusterer, predictions)


if __name__ == '__main__':
    # Configure Spark
    sc, spark = confugure_spark(APP_NAME)

    # Execute Main functionality
    main(sc, spark)
