# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
from vespa.application import Vespa
from vespa.evaluation import VespaFeatureCollector
import vespa.querybuilder as qb
from typing import Dict, Any
import json
import logging
from pathlib import Path
import argparse


def feature_collection_second_phase_query_fn(
    query_text: str, top_k: int = 10, query_id: str = None
) -> Dict[str, Any]:
    """
    Convert plain text into a JSON body for Vespa query with 'feature-collection' rank profile.
    Includes both semantic similarity and BM25 matching with match features.
    """
    return {
        "yql": str(
            qb.select("*")
            .from_("doc")
            .where(
                (
                    qb.nearestNeighbor(
                        field="title_embedding",
                        query_vector="embedding",
                        annotations={
                            "targetHits": 100,
                            "label": "title_label",
                        },
                    )
                    | qb.nearestNeighbor(
                        field="chunk_embeddings",
                        query_vector="embedding",
                        annotations={
                            "targetHits": 100,
                            "label": "chunk_label",
                        },
                    )
                    | qb.userQuery(
                        query_text,
                    )
                )
            )
        ),
        "query": query_text,
        "ranking": "collect-second-phase",
        "input.query(embedding)": f"embed({query_text})",
        "input.query(float_embedding)": f"embed({query_text})",
        "hits": top_k,
        "timeout": "5s",
        "presentation.summary": "no-chunks",
        "presentation.timing": True,
    }


def feature_collection_first_phase_query_fn(
    query_text: str, top_k: int = 10, query_id: str = None
) -> Dict[str, Any]:
    """
    Convert plain text into a JSON body for Vespa query with 'feature-collection' rank profile.
    Includes both semantic similarity and BM25 matching with match features.
    """
    return {
        "yql": str(
            qb.select("*")
            .from_("doc")
            .where(
                (
                    qb.nearestNeighbor(
                        field="title_embedding",
                        query_vector="embedding",
                        annotations={
                            "targetHits": 100,
                            "label": "title_label",
                        },
                    )
                    | qb.nearestNeighbor(
                        field="chunk_embeddings",
                        query_vector="embedding",
                        annotations={
                            "targetHits": 100,
                            "label": "chunk_label",
                        },
                    )
                    | qb.userQuery(
                        query_text,
                    )
                )
            )
        ),
        "query": query_text,
        "ranking": "collect-training-data",
        "input.query(embedding)": f"embed({query_text})",
        "input.query(float_embedding)": f"embed({query_text})",
        "hits": top_k,
        "timeout": "5s",
        "presentation.summary": "no-chunks",
        "presentation.timing": True,
    }


def generate_collector_name(
    collect_matchfeatures: bool,
    collect_rankfeatures: bool,
    collect_summaryfeatures: bool,
    second_phase: bool,
) -> str:
    """
    Generate a collector name based on feature collection settings and phase.

    Args:
        collect_matchfeatures: Whether match features are being collected
        collect_rankfeatures: Whether rank features are being collected
        collect_summaryfeatures: Whether summary features are being collected
        second_phase: Whether using second phase (True) or first phase (False)

    Returns:
        Generated collector name string
    """
    features = []
    if collect_matchfeatures:
        features.append("match")
    if collect_rankfeatures:
        features.append("rank")
    if collect_summaryfeatures:
        features.append("summary")

    features_str = "_".join(features) if features else "nofeatures"
    phase_str = "second_phase" if second_phase else "first_phase"
    return f"{features_str}_{phase_str}"


def main(args):
    dataset_path = Path(args.dataset_dir)
    queries_path = dataset_path / args.queries_filename

    # Validate file exists
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")

    logging.info(f"Loading queries from: {queries_path}")
    with open(queries_path, "r") as file:
        queries_data = json.load(file)

    ids_to_text = {q["query_id"]: q["query_text"] for q in queries_data}
    relevant_docs = {
        q["query_id"]: set(map(str, q["relevant_document_ids"])) for q in queries_data
    }

    # Generate collector_name if not provided
    if not args.collector_name:
        args.collector_name = generate_collector_name(
            args.collect_matchfeatures,
            args.collect_rankfeatures,
            args.collect_summaryfeatures,
            args.second_phase,
        )
        logging.info(f"Generated collector name: {args.collector_name}")

    logging.info(f"Connecting to Vespa at {args.vespa_url}:{args.vespa_port}")
    app = Vespa(url=args.vespa_url, port=args.vespa_port)
    function_to_use = (
        feature_collection_second_phase_query_fn
        if args.second_phase
        else feature_collection_first_phase_query_fn
    )
    feature_collector = VespaFeatureCollector(
        queries=ids_to_text,
        relevant_docs=relevant_docs,
        vespa_query_fn=function_to_use,
        app=app,
        name=args.collector_name,
        id_field="id",
        collect_matchfeatures=args.collect_matchfeatures,
        collect_summaryfeatures=args.collect_summaryfeatures,
        collect_rankfeatures=args.collect_rankfeatures,
        write_csv=True,
        random_hits_strategy="ratio",
        random_hits_value=1,
    )
    results = feature_collector.collect()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect training data from Vespa using pyvespa VespaFeatureCollector."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../queries",
        help="Directory containing the queries JSON file. Assumed relative to script location if not absolute.",
    )
    parser.add_argument(
        "--second_phase",
        action="store_true",
        default=False,
        help="Use second phase of feature collection. Else uses first phase.",
    )
    parser.add_argument(
        "--queries_filename",
        type=str,
        default="queries.json",
        help="Filename of the JSON file containing queries.",
    )
    parser.add_argument(
        "--vespa_url",
        type=str,
        default="http://localhost",
        help="Vespa application URL.",
    )
    parser.add_argument(
        "--vespa_port", type=int, default=8080, help="Vespa application port."
    )
    parser.add_argument(
        "--collector_name",
        type=str,
        required=False,
        help="Name for the VespaFeatureCollector instance.",
    )
    parser.add_argument(
        "--collect_matchfeatures",
        action="store_true",
        default=False,
        help="Collect match features from Vespa responses.",
    )
    parser.add_argument(
        "--collect_summaryfeatures",
        action="store_true",
        default=False,
        help="Collect summary features from Vespa responses.",
    )
    parser.add_argument(
        "--collect_rankfeatures",
        action="store_true",
        default=False,
        help="Collect rank features from Vespa responses.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
