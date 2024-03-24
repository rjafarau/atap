from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext


APP_NAME = "My Spark Application"


def confugure_spark(app_name):
    # Create SparkContext and SparkSession
    conf = (SparkConf()
            .setAppName(app_name)
            .setMaster('local[*]'))
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    spark.sparkContext.setLogLevel('WARN')
    return sc, spark


def main(sc):
    # Define RDDs and apply operations and actions to them.
    pass


if __name__ == "__main__":
    # Configure Spark
    sc, spark = confugure_spark(APP_NAME)

    # Execute Main functionality
    main(sc)
