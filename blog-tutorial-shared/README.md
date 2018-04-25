<!-- Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa tutorial utility scripts

This directory contains utility code for the blog-search and blog-recommendation sample applications.

## Vespa Tutorial pt. 1

### From raw JSON to Vespa Feeding format

    $ python parse.py trainPosts.json > somefile.json

Parses JSON from the file trainPosts.json downloaded from Kaggle during the [blog search tutorial](http://docs.vespa.ai/documentation/tutorials/blog-search.html) and format it according to Vespa Document JSON format.

    $ python parse.py -p trainPosts.json > somefile.json

Give it the flag "-p" or "--popularity", and the script also calculates and adds the field `popularity`, as introduced [in the tutorial](http://docs.vespa.ai/documentation/tutorials/blog-search.html#blog-popularity-signal).

## Vespa Tutorial pt. 2

### Building and running the Spark script for calculating latent factors

1. Install the latest version of [Apache Spark](http://spark.apache.org/) and [sbt](http://www.scala-sbt.org/download.html).

2. Clone this repository and build the Spark script with `sbt package` (in the root directory of this repo).

3. Use the resulting jar file when running spark jobs included in the tutorials.

## Vespa Tutorial pt.3

Pre-computed data used throughout the tutorial will be made available shortly.

### Create Training Dataset

After running the Spark jobs in part 2, you will have created a
`blog-job/user_item_cf_cv` directory under the `blog-recommendation` sample
app. In the `blog-recommendation` directory, run:

    $ cat blog-job/user_item_cf_cv/product_features/part-0000* > blog-job/user_item_cf_cv/product.json
    $ cat blog-job/user_item_cf_cv/user_features/part-0000* > blog-job/user_item_cf_cv/user.json
    $ cat blog-job/training_and_test_indices/training_set_ids/part-000* > blog-job/training_and_test_indices/train.txt

This creates the JSON files used to create the training set. Before running the
script to create the dataset, check that you have installed the dependencies
and created a output directory.

    $ mkdir blog_job/nn_model
    $ r

    ...

    > install.packages("jsonlite")
    > install.packages("dplyr")

Then, generate the dataset:

    $ r --vanilla < ../blog-tutorial-shared/src/R/generateDataset.R

The dataset will be put in `blog-job/nn_model/training_set.txt` which will be used next.

### Train model with TensorFlow

Train the model with

    $ python vespaModel.py --product_features_file_path vespa_tutorial_data/user_item_cf_cv/product.json \
                           --user_features_file_path vespa_tutorial_data/user_item_cf_cv/user.json \
                           --dataset_file_path vespa_tutorial_data/nn_model/training_set.txt

Model summary statistics will be saved at folder
```runs/${start_time}``` with ```${start_time}``` representing the time you
started to train the model.

Visualize the accuracy and loss metrics with

    $ tensorboard --logdir runs

### Export model parameters to Tensor Vespa format

The `vespaModel.py` script saves the model to a directory `saved` which can be
imported directly into Vespa. Please see [Ranking with TensorFlow models in
Vespa](http://docs.vespa.ai/documentation/tensorflow.html) for more details.

### Offline evaluation

Query Vespa using the rank-profile ```tensor``` for users in the test set and return 100 blog post recommendations. Use those recommendations in the information contained in the test set to compute
metrics defined in the Tutorial pt. 2.

    pig -x local -f tutorial_compute_metric.pig \
      -param VESPA_HADOOP_JAR=vespa-hadoop.jar \
      -param TEST_INDICES=blog-job/training_and_test_indices/testing_set_ids \
      -param ENDPOINT=$(hostname):8080
      -param NUMBER_RECOMMENDATIONS=100
      -param RANKING_NAME=tensor
      -param OUTPUT=blog-job/cf-metric

Repeat the process, but now using the rank-profile ```nn_tensor```.

    pig -x local -f tutorial_compute_metric.pig \
      -param VESPA_HADOOP_JAR=vespa-hadoop.jar \
      -param TEST_INDICES=blog-job/training_and_test_indices/testing_set_ids \
      -param ENDPOINT=$(hostname):8080
      -param NUMBER_RECOMMENDATIONS=100
      -param RANKING_NAME=nn_tensor
      -param OUTPUT=blog-job/cf-metric
