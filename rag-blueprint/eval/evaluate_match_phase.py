# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
from vespa.evaluation import VespaMatchEvaluator
from vespa.application import Vespa
import vespa.querybuilder as qb
import json
from pathlib import Path

SCHEMA_NAME = "doc"
VESPA_URL = "http://localhost:8080"
QUERY_FILE = Path(__file__).parent.parent / "queries" / "queries.json"


def match_weakand_query_fn(query_text: str, top_k: int) -> dict:
    return {
        "yql": str(qb.select("*").from_(SCHEMA_NAME).where(qb.userQuery(query_text))),
        "query": query_text,
        "ranking": "match-only",
        "input.query(embedding)": f"embed({query_text})",
        "presentation.summary": "no-chunks",
    }


def match_hybrid_query_fn(query_text: str, top_k: int) -> dict:
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
        "query": query_text,
        "ranking": "match-only",
        "input.query(embedding)": f"embed({query_text})",
        "presentation.summary": "no-chunks",
    }


def match_semantic_query_fn(query_text: str, top_k: int) -> dict:
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
            )
        ),
        "query": query_text,
        "ranking": "match-only",
        "input.query(embedding)": f"embed({query_text})",
        "presentation.summary": "no-chunks",
    }


app = Vespa(VESPA_URL)

with open(QUERY_FILE, "r") as f:
    queries = json.load(f)

ids_to_query = {query["query_id"]: query["query_text"] for query in queries}
relevant_docs = {
    query["query_id"]: set(query["relevant_document_ids"])
    for query in queries
    if "relevant_document_ids" in query
}

match_results = {}
for evaluator_name, query_fn in [
    ("semantic", match_semantic_query_fn),
    ("weakand", match_weakand_query_fn),
    ("hybrid", match_hybrid_query_fn),
]:
    print(f"Evaluating {evaluator_name}...")

    match_evaluator = VespaMatchEvaluator(
        queries=ids_to_query,
        relevant_docs=relevant_docs,
        vespa_query_fn=query_fn,
        app=app,
        name="test-run",
        id_field="id",
        write_csv=True,
        write_verbose=True,  # optionally write verbose metrics to CSV
    )

    results = match_evaluator()
    match_results[evaluator_name] = results
    print(f"Results for {evaluator_name}:")
    print(results)
