#! /usr/bin/env python3

import re
import json
import random
from typing import Tuple
import os.path
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import pandas as pd
import tensorflow as tf
import tensorflow_ranking as tfr
import keras_tuner as kt
import numpy as np

from vespa.package import (
    ApplicationPackage,
    Field,
    FieldSet,
    RankProfile as Ranking,
    QueryField,
)
from vespa.evaluation import NormalizedDiscountedCumulativeGain as NDCG
from vespa.deployment import VespaDocker
from vespa.application import Vespa
from vespa.query import QueryModel, RankProfile, WeakAnd, OR

REPLACE_SYMBOLS = ["(", ")", " -", " +"]
QUOTES = [
    "\u0022",  # quotation mark (")
    "\u0027",  # apostrophe (')
    "\u00ab",  # left-pointing double-angle quotation mark
    "\u00bb",  # right-pointing double-angle quotation mark
    "\u2018",  # left single quotation mark
    "\u2019",  # right single quotation mark
    "\u201a",  # single low-9 quotation mark
    "\u201b",  # single high-reversed-9 quotation mark
    "\u201c",  # left double quotation mark
    "\u201d",  # right double quotation mark
    "\u201e",  # double low-9 quotation mark
    "\u201f",  # double high-reversed-9 quotation mark
    "\u2039",  # single left-pointing angle quotation mark
    "\u203a",  # single right-pointing angle quotation mark
    "\u300c",  # left corner bracket
    "\u300d",  # right corner bracket
    "\u300e",  # left white corner bracket
    "\u300f",  # right white corner bracket
    "\u301d",  # reversed double prime quotation mark
    "\u301e",  # double prime quotation mark
    "\u301f",  # low double prime quotation mark
    "\ufe41",  # presentation form for vertical left corner bracket
    "\ufe42",  # presentation form for vertical right corner bracket
    "\ufe43",  # presentation form for vertical left corner white bracket
    "\ufe44",  # presentation form for vertical right corner white bracket
    "\uff02",  # fullwidth quotation mark
    "\uff07",  # fullwidth apostrophe
    "\uff62",  # halfwidth left corner bracket
    "\uff63",  # halfwidth right corner bracket
]
REPLACE_SYMBOLS.extend(QUOTES)


def download_and_unzip_dataset(data_dir: str, dataset_name: str) -> str:
    """
    Download and unzip dataset

    :param data_dir: Folder path to hold the downloaded files
    :param dataset_name: Name of the dataset according to BEIR benchmark

    :return: Return the path of the folder containing the unzipped dataset files.
    """
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset_name
    )
    data_path = util.download_and_unzip(url, data_dir)
    print("Dataset downloaded here: {}".format(data_path))
    return data_path


class SearchApp:
    def __init__(self, application_name):
        self.application_name = application_name

    def create_application_package(self):
        app_package = ApplicationPackage(name=self.application_name)
        app_package.schema.add_fields(
            Field(name="id", type="string", indexing=["attribute", "summary"]),
            Field(
                name="title",
                type="string",
                indexing=["index"],
                index="enable-bm25",
            ),
            Field(
                name="body",
                type="string",
                indexing=["index"],
                index="enable-bm25",
            ),
        )
        app_package.schema.add_field_set(FieldSet(name="default", fields=["body"]))
        app_package.schema.add_rank_profile(
            rank_profile=Ranking(
                name="bm25", first_phase="bm25(body)", summary_features=["bm25(body)"]
            )
        )
        app_package.schema.add_rank_profile(
            rank_profile=Ranking(name="native_rank", first_phase="nativeRank(body)")
        )
        app_package.schema.add_rank_profile(
            rank_profile=Ranking(name="random", first_phase="random")
        )
        app_package.query_profile.add_fields(QueryField(name="maxHits", value=10000))
        return app_package

    def deploy_package(self, app_package):
        vespa_docker = VespaDocker(port=8089)
        app = vespa_docker.deploy(application_package=app_package)
        return app

    @staticmethod
    def send_feed_batch(app, corpus):
        """
        Feed document corpus to a Vespa app.

        :param app: Vespa connection.
        :param corpus: Dict containing document id as key and a dict with document fields as the value.
        :return: List with feed response returned by pyvespa.
        """
        batch_feed = [
            {
                "id": idx,
                "fields": {
                    "id": idx,
                    "title": corpus[idx].get("title", None),
                    "body": corpus[idx].get("text", None),
                },
            }
            for idx in list(corpus.keys())
        ]
        feed_results = app.feed_batch(batch=batch_feed)
        return feed_results

    @staticmethod
    def collect_data(
        app, output_file, labeled_data, batch_size, query_model, number_additional_docs
    ):
        """
        Collect labeled data with Vespa rank features.

        :param app: Vespa connection to the app.
        :param output_file: File path to store the results
        :param labeled_data: pyvespa labeled data
        :param batch_size: Size of the batch to send for each request.
        :param query_model: Query model containing match phase and ranking phase used to collect non-relevant
            query-document pairs.
        :param number_additional_docs: The number of additional documents to collect for each labeled data point.
        :return: Data is stored in the output file and no object is returned.
        """
        labeled_data_batches = [
            labeled_data[i : i + batch_size]
            for i in range(0, len(labeled_data), batch_size)
        ]
        for idx, ld in enumerate(labeled_data_batches):
            training_data_batch = app.collect_training_data(
                labeled_data=ld,
                id_field="id",
                query_model=query_model,
                number_additional_docs=number_additional_docs,
                fields=["rankfeatures", "summaryfeatures"],
            )
            print("{}/{}".format(idx, len(labeled_data_batches)))
            if os.path.isfile(output_file):
                training_data_batch.to_csv(
                    path_or_buf=output_file, header=False, index=False, mode="a"
                )
            else:
                training_data_batch.to_csv(
                    path_or_buf=output_file, header=True, index=False, mode="w"
                )

    @staticmethod
    def online_evaluation(app, labeled_data, query_model):
        vespa_metric = app.evaluate(
            labeled_data=labeled_data,
            eval_metrics=[NDCG(at=10)],
            query_model=query_model,
            id_field="id",
        )
        return vespa_metric.loc["ndcg_10"].loc["mean", query_model.name]


class Dataset:
    def __init__(self):
        pass

    @staticmethod
    def prepare_data(data_path: str, split_type: str = "test") -> Tuple:
        """
        Extract corpus, queries and qrels from the dataset.

        :param data_path: Folder path that contains the unzipped dataset files.
        :param split_type: One of 'train', 'dev' or 'test' set.

        :return: a tuple containing 'corpus', 'queries' and 'qrels'.
        """
        corpus, queries, qrels = GenericDataLoader(data_path).load(
            split=split_type
        )  # or split = "train" or "dev"
        return corpus, queries, qrels

    @staticmethod
    def sample_data(
        corpus,
        train_qrels,
        train_queries,
        dev_qrels,
        dev_queries,
        number_positive_samples,
        number_negative_samples,
    ):
        """
        Sample from dataset following BEIR data format.

        :param corpus: Dict containing document id as key and document content as value.
        :param train_qrels: Dict containing training query id as key and a dict with doc_id:score as value.
        :param train_queries: Dict containing training query id as key and query string as value.
        :param dev_qrels: Dict containing dev query id as key and a dict with doc_id:score as value.
        :param dev_queries: Dict containing dev query id as key and query string as value.
        :param number_positive_samples: The number of positive (query_id, relevant_doc)-pairs to select for training
            and for dev. If number_positive_samples=100 it means we will sample 100 pairs
            from the training set and 100 pairs from the dev set. The relevant documents will be included in
            the document corpus.
        :param number_negative_samples: The number of documents to be randomly chosen from the document corpus, in
            addition to the relevant documents sampled.

        :return: Tuple with the following elements: corpus_sample, train_qrels_sample, train_queries_sample,
            dev_qrels_sample, dev_queries_sample.
        """

        train_qrels_sample = {
            k: train_qrels[k]
            for k in random.sample(
                k=number_positive_samples, population=sorted(train_qrels)
            )
        }
        dev_qrels_sample = {
            k: dev_qrels[k]
            for k in random.sample(
                k=number_positive_samples, population=sorted(dev_qrels)
            )
        }

        train_queries_sample = {k: train_queries[k] for k in train_qrels_sample.keys()}
        dev_queries_sample = {k: dev_queries[k] for k in dev_qrels_sample.keys()}

        train_positive_doc_id_samples = [
            doc_id[0]
            for doc_id in [list(docs.keys()) for docs in train_qrels_sample.values()]
        ]
        dev_positive_doc_id_samples = [
            doc_id[0]
            for doc_id in [list(docs.keys()) for docs in dev_qrels_sample.values()]
        ]

        negative_doc_id_samples = random.sample(
            k=number_negative_samples, population=sorted(corpus)
        )
        doc_id_samples = list(
            set(
                train_positive_doc_id_samples
                + dev_positive_doc_id_samples
                + negative_doc_id_samples
            )
        )
        corpus_sample = {k: corpus[k] for k in doc_id_samples}

        return (
            corpus_sample,
            train_qrels_sample,
            train_queries_sample,
            dev_qrels_sample,
            dev_queries_sample,
        )

    @staticmethod
    def replace_symbols(x):
        for symbol in REPLACE_SYMBOLS:
            x = x.replace(symbol, "")
        return x

    @staticmethod
    def parse_query(query):
        return re.sub(" +", " ", Dataset.replace_symbols(query)).strip()

    @staticmethod
    def create_labeled_data_from_beir_data(qrels, queries):
        """
        Create pyvespa labeled data from beir datasets

        :param qrels: Dict containing query id as key and a dict with doc_id:score as value.
        :param queries: Dict containing query id as key and query string as value.
        :return: pyvespa labeled data
        """
        labeled_data = [
            {
                "query_id": query_id,
                "query": Dataset.parse_query(queries[query_id]),
                "relevant_docs": [
                    {"id": relevant_doc_id, "score": qrels[query_id][relevant_doc_id]}
                    for relevant_doc_id in qrels[query_id].keys()
                ],
            }
            for query_id in qrels.keys()
        ]
        return labeled_data


class CollectedDataset:
    def __init__(self):
        pass

    @staticmethod
    def sample_collected_data(data_file, number_negatives, features):
        """
        Create a df from data collect from Vespa

        :param data_file: File path containing data collected from Vespa.
        :param number_negatives: Number of negatives to use. Sometimes we want to use fewer negatives than
            the maximum number of negatives originally collected.
        :param features: Which vespa ranking features to include in the df. Sometimes we want to use a subset of
            the features originally collected.
        :return: Data frame containing positive and nagative samples.
        """
        df = pd.read_csv(data_file)
        df = df[["document_id", "query_id", "label"] + features]
        df = df.drop_duplicates(["document_id", "query_id", "label"])
        df_positives = df[df["label"] == 1]
        df_negatives = df[df["label"] == 0]
        final_df = []
        for idx, row in df_positives.iterrows():
            final_df.append(
                pd.concat(
                    [
                        df_positives[df_positives["query_id"] == row["query_id"]],
                        df_negatives[df_negatives["query_id"] == row["query_id"]].head(
                            number_negatives
                        ),
                    ]
                )
            )
        return pd.concat(final_df, ignore_index=True)

    @staticmethod
    def create_train_and_dev_dfs(
        training_data_file, dev_data_file, number_negatives, features
    ):
        """
        Generate train and dev dfs with collected features

        :param training_data_file: File path of the training data collected.
        :param dev_data_file: File path of the dev data collected.
        :param number_negatives: Number of negatives to include in the dfs.
        :param features: Subset of the features to include in the dfs.
        :return: train and dev data frames.
        """
        train_df = CollectedDataset.sample_collected_data(
            data_file=training_data_file,
            number_negatives=number_negatives,
            features=features,
        )
        dev_df = CollectedDataset.sample_collected_data(
            data_file=dev_data_file,
            number_negatives=number_negatives,
            features=features,
        )
        return train_df, dev_df

    @staticmethod
    def create_labeled_data_from_df(df, queries):
        dev_labeled_data = []
        for idx, row in df[df["label"] == 1].iterrows():
            dev_labeled_data.append(
                {
                    "query_id": str(int(row["query_id"])),
                    "query": queries[str(int(row["query_id"]))],
                    "relevant_docs": [{"id": str(int(row["document_id"])), "score": 1}],
                }
            )
        return dev_labeled_data


class ListwiseLinearModel:
    def __init__(self):
        pass

    @staticmethod
    def keras_linear_model(number_documents_per_query):
        """
        Simple linear model with a sigmoid linear function for listwise prediction.

        :param number_documents_per_query: Number of documents per query to reshape the listwise prediction.
        :return: The uncompiled Keras model.
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    1,
                    use_bias=False,
                    activation=None,
                    kernel_constraint=tf.keras.constraints.NonNeg(),
                ),
                tf.keras.layers.Reshape((number_documents_per_query,)),
            ]
        )
        return model

    @staticmethod
    def keras_compiled_model(model, learning_rate):
        """
        Compile listwise Keras model with NDCG stateless metric and ApproxNDCGLoss

        :param model: uncompiled Keras model
        :param learning_rate: learning rate used in the Adagrad optim algo.
        :return: Keras compiled model.
        """
        ndcg = tfr.keras.metrics.NDCGMetric(topn=10)

        def ndcg_stateless(y_true, y_pred):
            ndcg.reset_states()
            return ndcg(y_true, y_pred)

        optimizer = tf.keras.optimizers.Adagrad(learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=tfr.keras.losses.ApproxNDCGLoss(),
            metrics=ndcg_stateless,
        )
        return model

    @staticmethod
    def tune_linear_model(
        number_documents_per_query,
        train_df,
        dev_df,
        feature_names,
        final_epoch,
        tune_early_stop_patience,
        tune_epochs,
        tuner_max_trials,
        tuner_dir,
        tuner_project_name,
    ):
        train_ds = ListwiseLinearModel.listwise_dataset_from_df(
            df=train_df,
            feature_names=feature_names,
            number_documents_per_query=number_documents_per_query,
        )
        dev_ds = ListwiseLinearModel.listwise_dataset_from_df(
            df=dev_df,
            feature_names=feature_names,
            number_documents_per_query=number_documents_per_query,
        )

        def build_model(hp):
            return ListwiseLinearModel.keras_compiled_model(
                model=ListwiseLinearModel.keras_linear_model(
                    number_documents_per_query=number_documents_per_query
                ),
                learning_rate=hp.Float(
                    "learning_rate", min_value=1e-2, max_value=1e2, sampling="log"
                ),
            )

        tuner = kt.RandomSearch(
            build_model,
            objective=kt.Objective("val_ndcg_stateless", direction="max"),
            directory=tuner_dir,
            project_name=tuner_project_name,
            overwrite=True,
            max_trials=tuner_max_trials,
            executions_per_trial=1,
        )
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_ndcg_stateless", patience=tune_early_stop_patience, mode="max"
        )
        tuner.search(
            train_ds.batch(32),
            validation_data=dev_ds.batch(32),
            epochs=tune_epochs,
            callbacks=[early_stopping_callback],
        )
        print(tuner.get_best_hyperparameters()[0].values)
        best_hps = tuner.get_best_hyperparameters()[0]
        model = build_model(best_hps)
        # early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        #     monitor="val_ndcg_stateless", patience=100, mode="max"
        # )
        model.fit(
            train_ds.batch(32),
            validation_data=dev_ds.batch(32),
            epochs=final_epoch,
            # callbacks=[early_stopping_callback],
        )
        weights = model.get_weights()
        eval_result_from_fit = model.history.history["val_ndcg_stateless"][-1]

        return weights, eval_result_from_fit

    @staticmethod
    def listwise_dataset_from_df(df, feature_names, number_documents_per_query):
        """
        Create TensorFlow dataframe suited for listwise loss function from pandas df.

        :param df: Pandas df containing the data.
        :param feature_names: Features to be used in the tensorflow model.
        :param number_documents_per_query: Number of documents per query. This will be used as the batch size
            of the TF dataset.
        :return: TF dataset
        """
        query_id_name = "query_id"
        target_name = "label"
        ds = tf.data.Dataset.from_tensor_slices(
            {
                "features": tf.cast(df[feature_names].values, tf.float32),
                "label": tf.cast(df[target_name].values, tf.float32),
                "query_id": tf.cast(df[query_id_name].values, tf.int64),
            }
        )

        key_func = lambda x: x[query_id_name]
        reduce_func = lambda key, dataset: dataset.batch(
            number_documents_per_query, drop_remainder=True
        )
        listwise_ds = ds.group_by_window(
            key_func=key_func,
            reduce_func=reduce_func,
            window_size=number_documents_per_query,
        )
        listwise_ds = listwise_ds.map(lambda x: (x["features"], x["label"]))
        return listwise_ds

    @staticmethod
    def offline_evaluation(
        train_df, dev_df, feature_names, number_documents_per_query, epochs=100
    ):
        """
        Offline evaluation of a simple linear model

        :param train_df: Training dataframe.
        :param dev_df: Dev dataframe, for evaluation
        :param feature_names: Feature names to be included.
        :param number_documents_per_query: number of documents per query
        :return: Tuple containing the compiled model and the final evaluation metric
        """
        train_ds = ListwiseLinearModel.listwise_dataset_from_df(
            df=train_df,
            feature_names=feature_names,
            number_documents_per_query=number_documents_per_query,
        )
        dev_ds = ListwiseLinearModel.listwise_dataset_from_df(
            df=dev_df,
            feature_names=feature_names,
            number_documents_per_query=number_documents_per_query,
        )
        model = ListwiseLinearModel.keras_linear_model(
            number_documents_per_query=number_documents_per_query
        )
        compiled_model = ListwiseLinearModel.keras_compiled_model(
            model=model, learning_rate=1
        )
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_ndcg_stateless", patience=100, mode="max"
        )
        compiled_model.fit(
            train_ds.batch(32),
            validation_data=dev_ds.batch(32),
            epochs=epochs,
            callbacks=[early_stopping_callback],
        )
        weights = compiled_model.get_weights()
        eval_result_from_fit = compiled_model.history.history["val_ndcg_stateless"][-1]

        return weights, eval_result_from_fit

    @staticmethod
    def cheap_offline_evaluation(dev_df, feature_names, number_documents_per_query):
        """
        Compute offline evaluation without training the model

        The code here is currently restricted to the simple case where we
        use one feature with parameter value equal to 1.

        :param dev_df: dev dataset used to compute the evaluation metric.
        :param feature_names: List of feature names. The code is currently
            restricted to only one feature.
        :param number_documents_per_query: Number of documents per query, as the number of documents
            across queries currently needs to be the same.
        :return: Evaluation metric, currently set to be NDCG@10.
        """
        dev_ds = ListwiseLinearModel.listwise_dataset_from_df(
            df=dev_df,
            feature_names=feature_names,
            number_documents_per_query=number_documents_per_query,
        )
        model = ListwiseLinearModel.keras_linear_model(
            number_documents_per_query=number_documents_per_query
        )
        compiled_model = ListwiseLinearModel.keras_compiled_model(
            model=model, learning_rate=0.1
        )
        compiled_model.build(input_shape=(32, number_documents_per_query, 1))
        compiled_model.layers[0].set_weights(weights=[np.array([[1]])])
        eval_result = compiled_model.evaluate(dev_ds.batch(32))
        return eval_result[1]


def sample_data(
    data_dir, number_positive_samples, number_negative_samples, output_file
):
    """
    Routine to sample smaller msmarco datasets for prototyping.

    :param data_dir: Folder containing the beir dataset.
    :param number_positive_samples: The number of positive (query_id, relevant_doc)-pairs to select for training
        and for dev. If number_positive_samples=100 it means we will sample 100 pairs
        from the training set and 100 pairs from the dev set. The relevant documents will be included in
        the document corpus.
    :param number_negative_samples: The number of documents to be randomly chosen from the document corpus, in
        addition to the relevant documents sampled.
    :param output_file: json output file, e.g. sample.json.

    :return: Dict containing the following keys: 'corpus', 'train_qrels', 'train_queries', 'dev_qrels', 'dev_queries'.
    """
    corpus, train_queries, train_qrels = Dataset.prepare_data(
        data_path=data_dir, split_type="train"
    )
    _, dev_queries, dev_qrels = Dataset.prepare_data(
        data_path=data_dir, split_type="dev"
    )
    (
        corpus_sample,
        train_qrels_sample,
        train_queries_sample,
        dev_qrels_sample,
        dev_queries_sample,
    ) = Dataset.sample_data(
        corpus=corpus,
        train_qrels=train_qrels,
        train_queries=train_queries,
        dev_qrels=dev_qrels,
        dev_queries=dev_queries,
        number_positive_samples=number_positive_samples,
        number_negative_samples=number_negative_samples,
    )
    sample = {
        "corpus": corpus_sample,
        "train_qrels": train_qrels_sample,
        "train_queries": train_queries_sample,
        "dev_qrels": dev_qrels_sample,
        "dev_queries": dev_queries_sample,
    }
    with open(output_file, "w") as f:
        json.dump(sample, f)
    return sample


def load_sampled_data(file):
    """
    Load sampled data.

    :param file: Path to the json file containing sampled data.
    :return: Dict containing the following keys: 'corpus', 'train_qrels', 'train_queries', 'dev_qrels', 'dev_queries'.
    """
    with open(file, "r") as f:
        data = json.load(f)
    return data


def create_application_package():
    """
    Create the application package used to collect data and evaluate ranking functions.

    :return: An instance of pyvespa ApplicationPackage.
    """
    application_name = "msmarco"

    app_package = ApplicationPackage(name=application_name)
    app_package.schema.add_fields(
        Field(name="id", type="string", indexing=["attribute", "summary"]),
        Field(
            name="title",
            type="string",
            indexing=["index"],
            index="enable-bm25",
        ),
        Field(
            name="body",
            type="string",
            indexing=["index"],
            index="enable-bm25",
        ),
    )
    app_package.schema.add_field_set(FieldSet(name="default", fields=["body"]))
    app_package.schema.add_rank_profile(
        rank_profile=Ranking(
            name="bm25", first_phase="bm25(body)", summary_features=["bm25(body)"]
        )
    )
    app_package.schema.add_rank_profile(
        rank_profile=Ranking(
            name="native_rank",
            first_phase="nativeRank",
            summary_features=["bm25(body)"],
        )
    )
    app_package.schema.add_rank_profile(
        rank_profile=Ranking(
            name="field_match_significance",
            first_phase="fieldMatch(body).significance",
        )
    )
    app_package.schema.add_rank_profile(
        rank_profile=Ranking(
            name="random", first_phase="random", summary_features=["bm25(body)"]
        )
    )
    app_package.query_profile.add_fields(QueryField(name="maxHits", value=10000))
    return app_package


def deploy_app_package(app_package, port):
    """
    Deploy the search app.

    :param app_package: Instance of the pyvespa ApplicationPackage.
    :param port: Port allocated to the Docker container.
    :return: Return connection to the Vespa app.
    """
    vespa_docker = VespaDocker(port=port)
    app = vespa_docker.deploy(application_package=app_package)
    return app


def feed_app(app, corpus):
    """
    Feed corpus to application.

    :param app: The Vespa connection to the app.
    :param corpus: Dict containing document id as key and a dict with document fields as the value.
    :return: Return connection to the Vespa app.
    """
    print(len(corpus))
    feed_output = SearchApp.send_feed_batch(app=app, corpus=corpus)
    print(len([x.status_code for x in feed_output if x.status_code == 200]))
    return 0


def collect_vespa_features(
    app,
    query_model,
    number_additional_docs,
    train_qrels,
    train_queries,
    dev_qrels,
    dev_queries,
    train_output_file="train_data_collected.csv",
    dev_output_file="dev_data_collected.csv",
):
    """
    Collect train and dev datasets containing Vespa ranking features.

    :param app: Vespa connection.
    :param query_model: Query model containing match phase and ranking phase used to collect non-relevant
        query-document pairs.
    :param number_additional_docs: The number of additional documents to collect for each labeled data point.
    :param train_qrels: Dict containing training query id as key and a dict with doc_id:score as value.
    :param train_queries: Dict containing training query id as key and query string as value.
    :param dev_qrels: Dict containing dev query id as key and a dict with doc_id:score as value.
    :param dev_queries: Dict containing dev query id as key and query string as value.
    :param train_output_file: File path to store training data collected.
    :param dev_output_file: File path to store dev data collected.
    :return: Data is stored in the output files and no object is returned.
    """
    train_labeled_data = Dataset.create_labeled_data_from_beir_data(
        qrels=train_qrels, queries=train_queries
    )
    dev_labeled_data = Dataset.create_labeled_data_from_beir_data(
        qrels=dev_qrels, queries=dev_queries
    )
    SearchApp.collect_data(
        app=app,
        output_file=train_output_file,
        labeled_data=train_labeled_data,
        batch_size=100,
        query_model=query_model,
        number_additional_docs=number_additional_docs,
    )
    SearchApp.collect_data(
        app=app,
        output_file=dev_output_file,
        labeled_data=dev_labeled_data,
        batch_size=100,
        query_model=query_model,
        number_additional_docs=number_additional_docs,
    )


def compare_offline_and_online_evaluation(
    app,
    collected_train_file,
    collected_dev_file,
    list_number_docs_per_query,
    sampled_data,
    output_file,
):
    """
    Compare offline and online evaluation metrics.

    :param app: Vespa connection.
    :param collected_train_file: File path containing training data collected from Vespa.
    :param collected_dev_file: File path containing dev data collected from Vespa.
    :param list_number_docs_per_query: List with number of documents per query to use when computing the
        offline evaluation.
    :param sampled_data: Sampled dataset containing dev query strings.
    :param output_file: File path to store results
    :return:
    """
    dev_queries = sampled_data["dev_queries"]
    result = []

    for number_documents_per_query in list_number_docs_per_query:
        train_df, dev_df = CollectedDataset.create_train_and_dev_dfs(
            training_data_file=collected_train_file,
            dev_data_file=collected_dev_file,
            number_negatives=number_documents_per_query - 1,
            features=["bm25(body)", "nativeRank", "fieldMatch(body).significance"],
        )
        bm25_offline_eval = ListwiseLinearModel.cheap_offline_evaluation(
            dev_df,
            feature_names=["bm25(body)"],
            number_documents_per_query=number_documents_per_query,
        )
        significance_offline_eval = ListwiseLinearModel.cheap_offline_evaluation(
            dev_df,
            feature_names=["fieldMatch(body).significance"],
            number_documents_per_query=number_documents_per_query,
        )
        native_offline_eval = ListwiseLinearModel.cheap_offline_evaluation(
            dev_df,
            feature_names=["nativeRank"],
            number_documents_per_query=number_documents_per_query,
        )
        dev_labeled_dataset = CollectedDataset.create_labeled_data_from_df(
            df=dev_df, queries=dev_queries
        )
        bm25_online_eval = SearchApp.online_evaluation(
            app=app,
            labeled_data=dev_labeled_dataset,
            query_model=QueryModel(
                name="bm25",
                match_phase=WeakAnd(hits=100),
                rank_profile=RankProfile(name="bm25", list_features=True),
            ),
        )
        significance_online_eval = SearchApp.online_evaluation(
            app=app,
            labeled_data=dev_labeled_dataset,
            query_model=QueryModel(
                name="significance",
                match_phase=WeakAnd(hits=100),
                rank_profile=RankProfile(name="field_match_significance", list_features=True),
            ),
        )
        native_online_eval = SearchApp.online_evaluation(
            app=app,
            labeled_data=dev_labeled_dataset,
            query_model=QueryModel(
                name="native_rank",
                match_phase=WeakAnd(hits=100),
                rank_profile=RankProfile(name="native_rank", list_features=True),
            ),
        )
        result.append(
            {
                "number_documents_per_query": number_documents_per_query,
                "bm25_offline_eval": bm25_offline_eval,
                "significance_offline_eval": significance_offline_eval,
                "native_offline_eval": native_offline_eval,
                "bm25_online_eval": bm25_online_eval,
                "significance_online_eval": significance_online_eval,
                "native_online_eval": native_online_eval,
            }
        )

    result_df = pd.DataFrame.from_records(result)
    if os.path.isfile(output_file):
        result_df.to_csv(path_or_buf=output_file, header=False, index=False, mode="a")
    else:
        result_df.to_csv(path_or_buf=output_file, header=True, index=False, mode="w")
    print(result)


def positive_constrained_linear_model_experiment(
    collected_train_file,
    collected_dev_file,
    feature_names,
    final_epoch,
    tune_early_stop_patience,
    tune_epochs,
    tuner_max_trials,
    tuner_dir,
    tuner_project_name,
    output_file,
):
    """
    Offline evaluation of positive-constrained linear models.

    :param collected_train_file: Path to .csv file containing training data collected from Vespa app.
    :param collected_dev_file: Path to .csv file containing dev data collected from Vespa app.
    :param feature_names: List of feature names.
    :param final_epoch: Number of epochs to run the final model after hyperparameter search.
    :param tune_early_stop_patience: Number of epochs to use as patience for early stopping callback.
    :param tune_epochs: Number of epochs to use while doing hyperparameter search.
    :param tuner_max_trials: Number of trials to use in the hyperparameter search.
    :param tuner_dir: Folder to store hyperparameter tuner info.
    :param tuner_project_name: Name of the project. Used to store tuner info in a specific folder.
    :param output_file: JSON File path to store experiment information.
    """
    number_documents_per_query = (
        100  # 100 showed promising results when comparing offline and online eval
    )
    train_df, dev_df = CollectedDataset.create_train_and_dev_dfs(
        training_data_file=collected_train_file,
        dev_data_file=collected_dev_file,
        number_negatives=number_documents_per_query - 1,
        features=feature_names,
    )
    try:
        with open(output_file, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []
    while len(feature_names) > 0:
        weights, offline_eval_from_fit = ListwiseLinearModel.tune_linear_model(
            number_documents_per_query=number_documents_per_query,
            train_df=train_df,
            dev_df=dev_df,
            feature_names=feature_names,
            final_epoch=final_epoch,
            tune_early_stop_patience=tune_early_stop_patience,
            tune_epochs=tune_epochs,
            tuner_max_trials=tuner_max_trials,
            tuner_dir=tuner_dir,
            tuner_project_name=tuner_project_name,
        )
        weights = {
            feature_name: float(weights[0][idx][0])
            for idx, feature_name in enumerate(feature_names)
        }
        print({k: round(weights[k], 2) for k in weights})
        print(offline_eval_from_fit)
        print("Number of features: {}".format(len(feature_names)))
        partial_result = {"metric": offline_eval_from_fit, "weights": weights}
        results.append(partial_result)
        with open(output_file, "w") as f:
            json.dump(results, f)
        worst_feature = min(weights, key=weights.get)
        feature_names = [x for x in feature_names if x != worst_feature]

    return 0
