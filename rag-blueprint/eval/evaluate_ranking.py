# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
from vespa.evaluation import VespaEvaluator
from vespa.application import Vespa
import vespa.querybuilder as qb
import json
from pathlib import Path
import argparse
import logging

SCHEMA_NAME = "doc"

# -8.617450
# +19.648338*match_avg_top_3_chunk_sim_scores
# +0.112750*match_avg_top_3_chunk_text_scores
# +0.095697*match_bm25(chunks)
# +0.034014*match_bm25(title)
# +11.116891*match_max_chunk_sim_scores
# +0.113867*match_max_chunk_text_scores

linear_params = {
    "input.query(intercept)": -8.617450,
    "input.query(bm25_chunks_param)": 0.095697,
    "input.query(bm25_title_param)": 0.034014,
    "input.query(avg_top_3_chunk_sim_scores_param)": 19.648338,
    "input.query(avg_top_3_chunk_text_scores_param)": 0.112750,
    "input.query(max_chunk_sim_scores_param)": 11.116891,
    "input.query(max_chunk_text_scores_param)": 0.113867,
}


def rank_first_phase_query_fn(query_text: str, top_k: int) -> dict:
    return {
        "yql": str(
            qb.select("*")
            .from_(SCHEMA_NAME)
            .where(
                qb.nearestNeighbor(
                    field="title_embedding",
                    query_vector="embedding",
                    annotations={"targetHits": 100},
                )
                | qb.nearestNeighbor(
                    field="chunk_embeddings",
                    query_vector="embedding",
                    annotations={"targetHits": 100},
                )
                | qb.userQuery(
                    query_text,
                )
            )
        ),
        "hits": top_k,
        "query": query_text,
        "ranking": "learned-linear-new",
        "input.query(embedding)": f"embed({query_text})",
        "input.query(float_embedding)": f"embed({query_text})",
        "presentation.summary": "no-chunks",
    } | linear_params


def rank_second_phase_query_fn(query_text: str, top_k: int) -> dict:
    return {
        "yql": str(
            qb.select("*")
            .from_(SCHEMA_NAME)
            .where(
                qb.nearestNeighbor(
                    field="title_embedding",
                    query_vector="embedding",
                    annotations={"targetHits": 100},
                )
                | qb.nearestNeighbor(
                    field="chunk_embeddings",
                    query_vector="embedding",
                    annotations={"targetHits": 100},
                )
                | qb.userQuery(
                    query_text,
                )
            )
        ),
        "hits": top_k,
        "query": query_text,
        "ranking": "second-with-gbdt",
        "input.query(embedding)": f"embed({query_text})",
        "input.query(float_embedding)": f"embed({query_text})",
        "presentation.summary": "no-chunks",
    }


def main(args):
    dataset_path = Path(args.dataset_dir)
    queries_path = dataset_path / args.queries_filename

    # Validate file exists
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")

    logging.info(f"Loading queries from: {queries_path}")
    with open(queries_path, "r") as f:
        queries = json.load(f)

    ids_to_query = {query["query_id"]: query["query_text"] for query in queries}
    relevant_docs = {
        query["query_id"]: set(query["relevant_document_ids"])
        for query in queries
        if "relevant_document_ids" in query
    }

    logging.info(f"Connecting to Vespa at {args.vespa_url}:{args.vespa_port}")
    app = Vespa(url=args.vespa_url, port=args.vespa_port)

    function_to_use = (
        rank_second_phase_query_fn if args.second_phase else rank_first_phase_query_fn
    )

    match_evaluator = VespaEvaluator(
        queries=ids_to_query,
        relevant_docs=relevant_docs,
        vespa_query_fn=function_to_use,
        id_field="id",
        app=app,
        name=args.evaluator_name,
        write_csv=args.write_csv,
        precision_recall_at_k=args.precision_recall_at_k,
    )

    results = match_evaluator()
    print(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ranking performance using VespaEvaluator."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../dataset",
        help="Directory containing the queries JSON file. Assumed relative to script location if not absolute.",
    )
    parser.add_argument(
        "--second_phase",
        action="store_true",
        default=False,
        help="Use second phase ranking. Else uses first phase with linear parameters.",
    )
    parser.add_argument(
        "--queries_filename",
        type=str,
        default="test_queries.json",
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
        "--evaluator_name",
        type=str,
        default="test-run",
        help="Name for the VespaEvaluator instance.",
    )
    parser.add_argument(
        "--write_csv",
        action="store_true",
        default=False,
        help="Write evaluation results to CSV file.",
    )
    parser.add_argument(
        "--precision_recall_at_k",
        type=int,
        nargs="+",
        default=[10, 20],
        help="List of k values for precision and recall calculation.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
