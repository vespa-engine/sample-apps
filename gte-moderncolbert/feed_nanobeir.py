#!/usr/bin/env python3
"""
Feed all NanoBEIR datasets to a Vespa application.

This script downloads NanoBEIR datasets from Hugging Face (zeta-alpha-ai)
and feeds the corpus documents and queries to a local Vespa application.

Usage:
    uv run feed_nanobeir.py [--vespa-url URL] [--datasets DATASET1,DATASET2,...] [--schema SCHEMA]

Example:
    uv run feed_nanobeir.py
    uv run feed_nanobeir.py --vespa-url http://localhost:8080
    uv run feed_nanobeir.py --datasets NanoClimateFEVER,NanoNQ
"""

import argparse
from typing import Iterator

from datasets import load_dataset
from vespa.application import Vespa
from vespa.io import VespaResponse


# All available NanoBEIR datasets from zeta-alpha-ai
NANOBEIR_DATASETS = [
    "NanoArguAna",
    "NanoClimateFEVER",
    "NanoDBPedia",
    "NanoFEVER",
    "NanoFiQA2018",
    "NanoHotpotQA",
    "NanoMSMARCO",
    "NanoNFCorpus",
    "NanoNQ",
    "NanoQuoraRetrieval",
    "NanoSCIDOCS",
    "NanoSciFact",
    "NanoTouche2020",
]


def load_nanobeir_corpus(dataset_name: str) -> Iterator[dict]:
    """
    Load corpus documents from a NanoBEIR dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'NanoClimateFEVER')

    Yields:
        Document dictionaries with 'id' and 'fields' for Vespa feeding
    """
    full_name = f"zeta-alpha-ai/{dataset_name}"
    print(f"Loading corpus from {full_name}...")
    corpus = load_dataset(full_name, "corpus", split="train")

    for doc in corpus:
        yield {
            "id": doc["_id"],
            "fields": {
                "doc_id": doc["_id"],
                "text": doc["text"],
                "dataset": dataset_name,
            },
        }


def load_nanobeir_queries(dataset_name: str) -> Iterator[dict]:
    """
    Load queries from a NanoBEIR dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'NanoClimateFEVER')

    Yields:
        Query dictionaries with 'id' and 'fields' for Vespa feeding
    """
    full_name = f"zeta-alpha-ai/{dataset_name}"
    print(f"Loading queries from {full_name}...")
    queries = load_dataset(full_name, "queries", split="train")

    for query in queries:
        yield {
            "id": query["_id"],
            "fields": {
                "query_id": query["_id"],
                "text": query["text"],
                "dataset": dataset_name,
            },
        }


def load_nanobeir_qrels(dataset_name: str) -> list[dict]:
    """
    Load query-document relevance judgments from a NanoBEIR dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'NanoClimateFEVER')

    Returns:
        List of qrel dictionaries with query-id and corpus-id
    """
    full_name = f"zeta-alpha-ai/{dataset_name}"
    print(f"Loading qrels from {full_name}...")
    qrels = load_dataset(full_name, "qrels", split="train")
    return [{"query_id": q["query-id"], "corpus_id": q["corpus-id"]} for q in qrels]


def feed_callback(response: VespaResponse, id: str):
    """Callback function to handle feed responses."""
    if not response.is_successful():
        print(f"Error feeding document {id}: {response.get_json()}")


def feed_documents_to_vespa(
    app: Vespa,
    documents: Iterator[dict],
    schema: str,
    namespace: str = "nanobeir",
) -> int:
    """
    Feed documents to Vespa application.

    Args:
        app: Vespa application instance
        documents: Iterator of document dictionaries
        schema: Vespa schema name
        namespace: Vespa namespace

    Returns:
        Number of documents fed
    """
    docs_list = list(documents)
    if not docs_list:
        return 0

    app.feed_iterable(
        iter=docs_list,
        schema=schema,
        namespace=namespace,
        callback=feed_callback,
    )
    return len(docs_list)


def feed_all_datasets(
    vespa_url: str = "http://localhost:8080",
    datasets: list[str] | None = None,
    corpus_schema: str = "doc",
    query_schema: str | None = None,
    feed_queries: bool = False,
) -> dict:
    """
    Feed all NanoBEIR datasets to Vespa.

    Args:
        vespa_url: URL of the Vespa application
        datasets: List of dataset names to feed (default: all)
        corpus_schema: Schema name for corpus documents
        query_schema: Schema name for queries (if feeding queries)
        feed_queries: Whether to also feed queries as documents

    Returns:
        Dictionary with statistics about the feeding process
    """
    if datasets is None:
        datasets = NANOBEIR_DATASETS

    print(f"Connecting to Vespa at {vespa_url}...")
    app = Vespa(url=vespa_url)

    # Check if Vespa is accessible
    try:
        status = app.get_application_status()
        print("Connected to Vespa application")
    except Exception as e:
        print(f"Failed to connect to Vespa: {e}")
        raise

    stats = {
        "datasets": {},
        "total_corpus_docs": 0,
        "total_queries": 0,
    }

    for dataset_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'=' * 60}")

        dataset_stats = {"corpus_docs": 0, "queries": 0, "qrels": 0}

        # Feed corpus documents
        try:
            corpus_docs = load_nanobeir_corpus(dataset_name)
            num_docs = feed_documents_to_vespa(
                app, corpus_docs, schema=corpus_schema, namespace="nanobeir"
            )
            dataset_stats["corpus_docs"] = num_docs
            stats["total_corpus_docs"] += num_docs
            print(f"Fed {num_docs} corpus documents")
        except Exception as e:
            print(f"Error feeding corpus for {dataset_name}: {e}")

        # Optionally feed queries
        if feed_queries and query_schema:
            try:
                queries = load_nanobeir_queries(dataset_name)
                num_queries = feed_documents_to_vespa(
                    app, queries, schema=query_schema, namespace="nanobeir"
                )
                dataset_stats["queries"] = num_queries
                stats["total_queries"] += num_queries
                print(f"Fed {num_queries} queries")
            except Exception as e:
                print(f"Error feeding queries for {dataset_name}: {e}")

        # Load qrels for reference (not fed, but counted)
        try:
            qrels = load_nanobeir_qrels(dataset_name)
            dataset_stats["qrels"] = len(qrels)
        except Exception as e:
            print(f"Error loading qrels for {dataset_name}: {e}")

        stats["datasets"][dataset_name] = dataset_stats

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Feed NanoBEIR datasets to a Vespa application"
    )
    parser.add_argument(
        "--vespa-url",
        default="http://localhost:8080",
        help="URL of the Vespa application (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=f"Comma-separated list of datasets to feed (default: all). Available: {', '.join(NANOBEIR_DATASETS)}",
    )
    parser.add_argument(
        "--schema",
        default="doc",
        help="Vespa schema name for corpus documents (default: doc)",
    )
    parser.add_argument(
        "--query-schema",
        default=None,
        help="Vespa schema name for queries (if specified, queries will also be fed)",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available NanoBEIR datasets and exit",
    )

    args = parser.parse_args()

    if args.list_datasets:
        print("Available NanoBEIR datasets:")
        for ds in NANOBEIR_DATASETS:
            print(f"  - {ds}")
        return

    datasets = None
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
        # Validate dataset names
        for ds in datasets:
            if ds not in NANOBEIR_DATASETS:
                print(f"Warning: Unknown dataset '{ds}'. Available datasets:")
                for available_ds in NANOBEIR_DATASETS:
                    print(f"  - {available_ds}")
                return

    feed_queries = args.query_schema is not None

    print("NanoBEIR Dataset Feeder for Vespa")
    print("=" * 60)
    print(f"Vespa URL: {args.vespa_url}")
    print(f"Corpus Schema: {args.schema}")
    print(f"Datasets: {datasets or 'all'}")
    if feed_queries:
        print(f"Query Schema: {args.query_schema}")
    print("=" * 60)

    stats = feed_all_datasets(
        vespa_url=args.vespa_url,
        datasets=datasets,
        corpus_schema=args.schema,
        query_schema=args.query_schema,
        feed_queries=feed_queries,
    )

    print("\n" + "=" * 60)
    print("Feeding Complete!")
    print("=" * 60)
    print(f"Total corpus documents fed: {stats['total_corpus_docs']}")
    if feed_queries:
        print(f"Total queries fed: {stats['total_queries']}")
    print("\nPer-dataset statistics:")
    for ds_name, ds_stats in stats["datasets"].items():
        print(f"  {ds_name}:")
        print(f"    - Corpus docs: {ds_stats['corpus_docs']}")
        if feed_queries:
            print(f"    - Queries: {ds_stats['queries']}")
        print(f"    - Qrels: {ds_stats['qrels']}")


if __name__ == "__main__":
    main()
