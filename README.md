# NLPDocumentClustering
### Instructions to Run

#### Prerequisites:
Spark v2.1.1 is used higher needs to be installed in the system
IPython Shell for running document clustering
ENV variable SPARK_HOME should point to home directory of downloaded spark

##### Run following commands to start with document clustering...
```
$ sudo apt-get install ipython
$ cd $SPARK_HOME
```

##### Start pyspark using IPython Driver
```
$ PYSPARK_DRIVER_PYTHON=ipython ./spark/bin/pyspark
```

#### Under pyspark terminal, run Document Clustering, using K-Means clustering:

##### Use Word2Vec estimator for building document-term matrix
```
%run doc_clustering.py --input-file NLP-test.json --feature-extractor=Word2Vec --total-clusters=20 --max-iterations=20
```

##### Use TF-IDF estimator for building Term Frequency-Inverse Document Frequencey matrix
```
%run doc_clustering.py --input-file NLP-test.json --feature-extractor=TFIDF --total-clusters=20 --max-iterations=20
```
