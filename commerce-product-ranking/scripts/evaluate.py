# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pandas>=2.2.3",
#     "pyarrow>=19.0.0",
#     "pyvespa>=0.53.0",
# ]
# ///


import argparse
from typing import Any, Dict
import pandas as pd

from vespa.application import Vespa
from vespa.evaluation import VespaEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        type=str,
        required=True,
        help="Vespa endpoint, e.g. http://localhost",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Vespa port (default: 8080)"
    )
    parser.add_argument(
        "--ranking", type=str, required=True, help="Ranking profile to use"
    )
    parser.add_argument(
        "--example_file", type=str, required=True, help="Path to example parquet file"
    )
    parser.add_argument(
        "--qrel_file", type=str, required=True, help="Path to qrel file"
    )
    parser.add_argument(
        "--hits", type=int, default=400, help="Number of hits to return"
    )
    parser.add_argument(
        "--certificate", type=str, help="Path to SSL certificate (optional)"
    )
    parser.add_argument("--key", type=str, help="Path to SSL key (optional)")
    args = parser.parse_args()
    print(args)

    # ----------------------
    # Load and process examples
    # ----------------------
    df = pd.read_parquet(args.example_file)
    df = df[df["split"] == "test"]
    df = df[df["product_locale"] == "us"]
    df = df[df["small_version"] == 1]

    # Group by query_id to aggregate the query text and build the recall string
    grouped_df = (
        df.groupby("query_id")
        .agg(
            {
                "query": "first",
                "product_id": lambda ids: "+({})".format(
                    " ".join("id:{}".format(pid) for pid in ids)
                ),
            }
        )
        .reset_index()
        .rename(columns={"product_id": "recall"})
    )
    # Ensure query IDs are strings
    grouped_df["query_id"] = grouped_df["query_id"].astype(str)

    # Create dictionaries mapping query_id -> query text and query_id -> recall string
    query_dict = grouped_df.set_index("query_id")["query"].to_dict()
    qid_to_recall_str = grouped_df.set_index("query_id")["recall"].to_dict()

    # ----------------------
    # Load qrels from the Vespa Cloud sample
    # ----------------------
    df_qrels = pd.read_csv(
        args.qrel_file,
        sep=" ",
        header=None,
    )
    df_qrels.columns = ["query_id", "Q0", "product_id", "relevance"]
    # Mapping as in the pyvespa example: 1 -> 0, 2 -> 0.01, 3 -> 0.1, 4 -> 1
    esci_mapping = {1: 0, 2: 0.01, 3: 0.1, 4: 1}
    df_qrels["relevance"] = df_qrels["relevance"].map(esci_mapping)
    df_qrels["query_id"] = df_qrels["query_id"].astype(str)
    grouped_qrels = (
        df_qrels.groupby("query_id")
        .apply(lambda group: dict(zip(group["product_id"], group["relevance"])), include_groups=False)
        .to_dict()
    )

    # Only use queries that have associated qrels
    common_query_ids = set(query_dict.keys()) & set(grouped_qrels.keys())
    if not common_query_ids:
        print("No common query IDs found between examples and qrels.")
        return

    filtered_queries = {qid: query_dict[qid] for qid in common_query_ids}
    filtered_qrels = {qid: grouped_qrels[qid] for qid in common_query_ids}

    # ----------------------
    # Define the Vespa query function
    # ----------------------
    def vespa_query_fn(query_text: str, top_k: int, query_id: str) -> Dict[str, Any]:
        """
        Given a query text and query_id, return a Vespa query request body.
        The ranking profile and number of hits are provided as command-line arguments.
        The recall string is built from the qid_to_recall_str mapping.
        """
        return {
            "yql": (
                "select id from product where userQuery() or "
                "({targetHits:100}nearestNeighbor(title_embedding, q_title)) or "
                "({targetHits:100}nearestNeighbor(description_embedding, q_description))"
            ),
            "query": query_text,
            "input.query(q_title)": f'embed(title, "{query_text}")',
            "input.query(q_description)": f'embed(description, "{query_text}")',
            "input.query(query_tokens)": f'embed(tokenizer, "{query_text}")',
            "ranking": args.ranking,
            "hits": args.hits,
            "timeout": "15s",
            "recall": qid_to_recall_str[query_id],
            "ranking.softtimeout.enable": "false",
        }

    # ----------------------
    # Initialize the Vespa application using the provided endpoint and session
    # ----------------------
    # Note: The endpoint is expected to be something like "http://localhost"
    app = Vespa(args.endpoint, port=args.port, key=args.key, cert=args.certificate)

    # ----------------------
    # Create and run the evaluator
    # ----------------------
    # For evaluation, we set ndcg_at_k to the maximum number of relevant documents among all queries.
    ndcg_at_k_value = max(len(rels) for rels in filtered_qrels.values())
    evaluator = VespaEvaluator(
        queries=filtered_queries,
        relevant_docs=filtered_qrels,
        vespa_query_fn=vespa_query_fn,
        app=app,
        ndcg_at_k=[ndcg_at_k_value],
    )

    results = evaluator.run()

    # ----------------------
    # Output the evaluation results
    # ----------------------
    print("Evaluation results:")
    print(results)


if __name__ == "__main__":
    main()
