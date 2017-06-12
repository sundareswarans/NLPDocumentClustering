import os
import sys
import argparse
import json

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, IDF, RegexTokenizer
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.types import Row
from pyspark.ml.clustering import KMeans, KMeansModel
from w3lib.html import remove_tags


Algorithm="""
. load_document(): 			=> Load the JSON file, and yield pyspark.sql.types.Row objects
					   Row with documentIndexID, and text (\n stripped)
. create_dataframe():			=> Creates dataframe from load_document()
. create_pipeline():			=> Creates pyspark.ml.Pipeline with several stages in text processing
.     create_regex_tokenizer		=> use \\W pattern
.     create_stopwords		        => Removes less important words like 'if', 'the', 'of',...
.     create_tfidf_estimator		=> Use CountVectorizer + IDF
.     create_word2vec_estimator         => Use Word2Vec estimator, if selected
.     create_kmeans_clustering          => K-means clustering, with user provided number of clusters
.     use JacardMeasure			=> For proximity measure
      or Cosine Measure			=>       -do-
      Specify Objective function	=> Centroids are not changed from previous iteration
"""


INTERESTED_FIELDS = {
    'abstract', 
    'prospectkeyword', 
    'headline', 
    'entityname', 
    'searchabletext',  #=> Remove HTML Tags
    'twitterfollowers', 
    'fulltext', 
    'username',
    'displayname', 
    'twitterposts', 
    'mediatype', 
    'articletype', 
    'sourcename', 
    'country', 
    'datecreated', 
    'programname', 
}


def load_json_document(doc_path):
    """
    Loads the JSON document and yields
    doc_id and doc_text from from the interested fields
    """
    with open(doc_path, 'rb') as fp:
        nlp_data = json.load(fp, encoding='utf-8')

        # for each 'hits', extract the document text
        for doc in nlp_data['hits']:
            doc_id, doc_source = doc['_id'], doc['_source']
            doc_text = ''
            for key in INTERESTED_FIELDS:
                if key not in doc_source or doc_source[key] is None:
                    continue
                if key == 'searchabletext':
                    v = remove_tags(doc_source[key].encode('utf-8'))
                else:
                    v = doc_source[key]
                if not isinstance(v, str):
                    try:
                        v = str(v)
                    except UnicodeEncodeError:
                        v = v.encode('utf-8')
                doc_text = " ".join([doc_text, v])

            # remove \r\n characters
            doc_text = "".join(unicode(doc_text, 'utf-8').splitlines())

            # yields a spark SQL Row object
            yield Row(doc_id=doc_id, doc_text=doc_text)


def create_dataframe_from(spark, doc_path):
    """
    Creates data frame by loading the json document
    """
    fields = ['doc_id', 'doc_text']
    return spark.createDataFrame(load_json_document(doc_path), fields)


def create_kmeans_pipeline_from(data_frame,
                                feature_extractor,
                                total_clusters,
                                max_iterations):
    """
    Creates a pipeline for tokenizing the document to words,
    removing stop words, and add TF-IDF, and finally does
    clustering using k-means algorithm
    """
    tokenizer_transformer = RegexTokenizer(inputCol="doc_text",
                                           outputCol="words",
                                           pattern="\\W")

    stop_words_transformer = StopWordsRemover(inputCol="words",
                                              outputCol="filtered_words")

    pipeline_stages = [tokenizer_transformer, stop_words_transformer,]

    if feature_extractor == 'TFIDF':

        # create TF counter using CountVectorizer
        tf_estimator = CountVectorizer(inputCol="filtered_words", outputCol="TF")

        # create inverse-document-frequency counter
        idf_estimator = IDF(inputCol="TF", outputCol="features")

        # add them to the pipeline stages
        pipeline_stages.extend([tf_estimator, idf_estimator])

    elif feature_extractor == 'Word2Vec':

        # create word2vec feature extractor
        w2v_estimator = Word2Vec(inputCol="filtered_words", outputCol="features")
        
        # add this to pipeline stage
        pipeline_stages.append(w2v_estimator)
    else:
        raise ValueError('Unknown feature extractor:' % feature_extractor)

    # create KMeans clustering
    kmeans = KMeans(k=total_clusters,
                    featuresCol="features",
                    predictionCol="DocumentClass",
                    seed=1,
                    maxIter=max_iterations)

    # finally add Kmeans to the pipeline
    # which takes "features" output from the previous stage and 
    # does the prediction.
    # NOTE:
    # For document clustering cosine_similarity measure is the preferred one.
    # This pipeline uses SSE method
    #
    pipeline_stages.append(kmeans)
    return Pipeline(stages=pipeline_stages).fit(data_frame)


def main(argv=sys.argv[1:]):
   parser = argparse.ArgumentParser()
   parser.add_argument('-i', '--input-file',
                       help="Input NLP data",
                       required=True)
   parser.add_argument('-c', '--clustering-algo',
                       help="Clustering Algorithm",
                       default='KMeans')
   parser.add_argument('-f', '--feature-extractor',
                       choices=['TFIDF', 'Word2Vec', ],
                       help='Whether to use TermFreq-InverseDocFreq or'
                            'Word2Vec estimator',
                       default='Word2Vec')
   parser.add_argument('-t', '--total-clusters',
                       help='Total number of clusters',
                       type=int, default=20)
   parser.add_argument('-m', '--max-iterations',
                       help='Max number of iterations to perform',
                       type=int,
                       default=20)

   # parse the arguments
   args = parser.parse_args(argv)

   # create a SparkSession for running or clustering process
   spark = SparkSession.builder.appName('DocumentClustering').getOrCreate()

   # create data frame from the input file
   df = create_dataframe_from(spark, args.input_file)

   # create a pipeline model, that has the necessary
   # feature transformers, estimators, and clustering algorithmns
   # added to the stages in the expected order
   # of execution
   pipeline_model = create_kmeans_pipeline_from(data_frame=df,
                                                feature_extractor=args.feature_extractor,
                                                total_clusters=args.total_clusters,
                                                max_iterations=args.max_iterations
                                                )
   # applies all the transformations and estimators in the 
   # pipeline stage
   results = pipeline_model.transform(df)
   results.cache()

   # Display count of documents under each cluster class
   print results.select('DocumentClass')\
           .groupBy('DocumentClass')\
           .count().\
           orderBy('DocumentClass').\
           show(n=args.total_clusters)


if __name__ == '__main__':
    main(sys.argv[1:])

