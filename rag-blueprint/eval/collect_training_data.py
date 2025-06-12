import argparse
import json
from pathlib import Path
import pandas as pd
from vespa.application import Vespa
import vespa.querybuilder as qb
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def build_query_body(
    query_text: str,
    rank_profile: str,
    schema_name: str,
    relevant_doc_ids: set,
    get_relevant: bool,
    hits_count: int,
):
    """
    Build a Vespa query body.
    If `get_relevant` is True, it filters for relevant document IDs.
    If `get_relevant` is False, it excludes them and gets other documents.
    """
    recall_str = " ".join(["id:" + str(doc) for doc in relevant_doc_ids])
    if get_relevant:
        recall_str = f"+({recall_str})"
    else:
        recall_str = f"-({recall_str})"
    return {
        "yql": str(
            qb.select("*")
            .from_(schema_name)
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
        "ranking": rank_profile,
        "hits": hits_count,
        "timeout": "5s",  # Increased timeout for potentially larger queries
        "input.query(embedding)": f"embed({query_text})",
        "recall": recall_str,
    }


def extract_features_from_response(response):
    """
    Extract features from the Vespa response.
    Yields a dictionary for each hit containing docid and matchfeatures.
    """
    hits = response.hits
    if not hits:
        logging.warning(
            f"No hits found in response for query: {response.get_json().get('root', {}).get('fields', {}).get('query')}"
        )
    for hit in hits:
        fields = hit.get("fields", {})
        doc_id = fields.get("id")
        match_features = fields.get("matchfeatures", {})
        if doc_id is not None:  # Ensure doc_id is present
            yield {
                "docid": str(doc_id),
                **match_features,
            }
        else:
            logging.warning(f"Hit without docid: {hit}")


def main(args):
    dataset_path = Path(args.dataset_dir)
    queries_path = dataset_path / args.queries_filename

    logging.info(f"Loading queries from: {queries_path}")
    with open(queries_path, "r") as file:
        queries_data = json.load(file)

    ids_to_text = {q["query_id"]: q["query_text"] for q in queries_data}
    relevant_docs_original = {
        q["query_id"]: set(map(str, q["relevant_document_ids"])) for q in queries_data
    }

    logging.info(f"Connecting to Vespa at {args.vespa_url}:{args.vespa_port}")
    app = Vespa(url=args.vespa_url, port=args.vespa_port)
    try:
        app.get_application_status()
        logging.info("Successfully connected to Vespa application.")
    except Exception as e:
        logging.error(f"Failed to connect to Vespa: {e}")
        return

    bodies = []
    query_metadata_list = []
    is_relevant_flags = (True, False)

    logging.info("Building query bodies...")
    for qid, q_text in ids_to_text.items():
        current_relevant_doc_ids = relevant_docs_original.get(qid, set())
        num_relevant = len(current_relevant_doc_ids)

        for get_relevant_flag in is_relevant_flags:
            # Determine how many hits to request
            # If get_relevant_flag is True, request all relevant docs
            # If get_relevant_flag is False, request the same number of non-relevant docs (or a fixed number if no relevant docs)
            hits_to_request = (
                num_relevant if num_relevant > 0 else args.default_hits_for_negatives
            )
            if (
                not get_relevant_flag and num_relevant == 0
            ):  # No relevant docs to base count on for negatives
                hits_to_request = args.default_hits_for_negatives
            elif not get_relevant_flag and num_relevant > 0:
                hits_to_request = num_relevant

            if not current_relevant_doc_ids and get_relevant_flag:
                logging.warning(
                    f"Query ID {qid} has no relevant documents specified. Skipping 'relevant' part for this query."
                )
                continue  # Skip trying to fetch relevant if none are listed

            bodies.append(
                build_query_body(
                    q_text,
                    args.rank_profile,
                    args.schema_name,
                    current_relevant_doc_ids,
                    get_relevant=get_relevant_flag,
                    hits_count=hits_to_request,
                )
            )
            query_metadata_list.append(
                {
                    "query_id": qid,
                    "query_text": q_text,
                    "relevant": get_relevant_flag,  # Store if we intended to get relevant docs
                }
            )
    if not bodies:
        logging.error("No query bodies were generated. Exiting.")
        return

    logging.info(
        f"Sending {len(bodies)} queries to Vespa in batches of {args.batch_size}..."
    )
    responses = app.query_many(queries=bodies, batch_size=args.batch_size)

    rows = []
    logging.info("Extracting features from responses...")
    for response, query_meta in zip(responses, query_metadata_list):
        if response.is_successful():
            for doc_feats in extract_features_from_response(response):
                rows.append(
                    {
                        **doc_feats,
                        **query_meta,
                    }
                )
        else:
            logging.error(
                f"Query failed: {query_meta['query_text']}. Response: {response.get_json()}"
            )

    if not rows:
        logging.error(
            "No data rows were extracted. Cannot proceed to create DataFrame or save CSV."
        )
        return

    df = pd.DataFrame(rows)
    logging.info(f"Collected {len(df)} feature rows.")

    # Sanity check
    # Reconstruct relevant_docs from the DataFrame where actual_relevance_label is True
    if (
        not df.empty
        and "relevant" in df.columns
        and "query_id" in df.columns
        and "docid" in df.columns
    ):
        relevant_docs_from_df = (
            df[df["relevant"]]
            .groupby("query_id")["docid"]
            .apply(lambda x: set(map(str, x)))  # Ensure docid is string for comparison
            .to_dict()
        )

        # Compare ensuring all keys in original are present in df-derived, and sets match
        match = True
        for qid, doc_ids_original_set in relevant_docs_original.items():
            if (
                not doc_ids_original_set
            ):  # Skip if no relevant docs were defined for this query
                continue
            doc_ids_df_set = relevant_docs_from_df.get(qid, set())
            if doc_ids_original_set != doc_ids_df_set:
                logging.warning(
                    f"Sanity check failed for query_id {qid}: "
                    f"Original: {doc_ids_original_set}, From DF: {doc_ids_df_set}"
                )
                match = False
        if match:
            logging.info(
                "Sanity check passed: Relevant documents from DataFrame match original relevant documents."
            )
        else:
            logging.warning(
                "Sanity check failed: Discrepancies found. See warnings above."
            )
    else:
        logging.warning(
            "DataFrame is empty or missing required columns for sanity check.",
            f"Columns present: {df.columns.tolist() if not df.empty else 'None'}",
        )

    if args.csv_output_dir:
        output_dir = Path(args.csv_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / args.output_filename
        df.to_csv(output_file_path, index=False)
        logging.info(f"Training data saved to {output_file_path}")
    else:
        logging.info("CSV output directory not specified. Skipping saving DataFrame.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect training data from Vespa for a learning-to-rank model."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../dataset",
        help="Directory containing the queries JSON file. Assumed relative to script location if not absolute.",
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
        "--schema_name", type=str, default="doc", help="Vespa schema name to query."
    )
    parser.add_argument(
        "--rank_profile",
        type=str,
        default="collect-training-data",
        help="Vespa rank profile to use for collecting features.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,  # Reduced default batch size for stability with many queries
        help="Batch size for sending queries to Vespa using query_many.",
    )
    parser.add_argument(
        "--csv_output_dir",
        type=str,
        default="collected_training_data",
        help="Directory to save the output CSV file. If not provided, CSV is not saved.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="training_features.csv",
        help="Name of the output CSV file.",
    )
    parser.add_argument(
        "--default_hits_for_negatives",
        type=int,
        default=10,  # Number of negative samples to fetch if a query has no positive examples
        help="Default number of hits to request for negative samples if a query has no positive examples.",
    )

    args = parser.parse_args()
    main(args)
