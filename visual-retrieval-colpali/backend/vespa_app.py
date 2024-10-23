import os
from vespa.application import Vespa
from vespa.io import VespaQueryResponse
from dotenv import load_dotenv
import torch
import numpy as np
import json
import time
from typing import Dict, Any, Tuple

MAX_QUERY_TERMS = 64


def get_vespa_app():
    load_dotenv()
    vespa_app_url = os.environ.get(
        "VESPA_APP_URL"
    )  # Ensure this is set to your Vespa app URL
    vespa_cloud_secret_token = os.environ.get("VESPA_CLOUD_SECRET_TOKEN")

    if not vespa_app_url or not vespa_cloud_secret_token:
        raise ValueError(
            "Please set the VESPA_APP_URL and VESPA_CLOUD_SECRET_TOKEN environment variables"
        )
    # Instantiate Vespa connection
    vespa_app = Vespa(
        url=vespa_app_url, vespa_cloud_secret_token=vespa_cloud_secret_token
    )
    vespa_app.wait_for_application_up()
    print(f"Connected to Vespa at {vespa_app_url}")
    return vespa_app


def format_query_results(query, response, hits=5) -> dict:
    query_time = response.json.get("timing", {}).get("searchtime", -1)
    query_time = round(query_time, 2)
    count = response.json.get("root", {}).get("fields", {}).get("totalCount", 0)
    result_text = f"Query text: '{query}', query time {query_time}s, count={count}, top results:\n"
    print(result_text)
    return response.json


async def query_vespa_default(
    app: Vespa,
    query: str,
    q_emb: torch.Tensor,
    hits: int = 3,
    timeout: str = "10s",
    **kwargs,
) -> dict:
    async with app.asyncio(connections=1) as session:
        query_embedding = format_q_embs(q_emb)

        start = time.perf_counter()
        response: VespaQueryResponse = await session.query(
            body={
                "yql": "select id,title,url,blur_image,page_number,snippet,text,summaryfeatures from pdf_page where userQuery();",
                "ranking": "default",
                "query": query,
                "timeout": timeout,
                "hits": hits,
                "input.query(qt)": query_embedding,
                "presentation.timing": True,
                **kwargs,
            },
        )
        assert response.is_successful(), response.json
        stop = time.perf_counter()
        print(
            f"Query time + data transfer took: {stop - start} s, vespa said searchtime was {response.json.get('timing', {}).get('searchtime', -1)} s"
        )
        open("response.json", "w").write(json.dumps(response.json))
    return format_query_results(query, response)


async def query_vespa_bm25(
    app: Vespa,
    query: str,
    q_emb: torch.Tensor,
    hits: int = 3,
    timeout: str = "10s",
    **kwargs,
) -> dict:
    async with app.asyncio(connections=1) as session:
        query_embedding = format_q_embs(q_emb)

        start = time.perf_counter()
        response: VespaQueryResponse = await session.query(
            body={
                "yql": "select id,title,url,blur_image,page_number,snippet,text,summaryfeatures from pdf_page where userQuery();",
                "ranking": "bm25",
                "query": query,
                "timeout": timeout,
                "hits": hits,
                "input.query(qt)": query_embedding,
                "presentation.timing": True,
                **kwargs,
            },
        )
        assert response.is_successful(), response.json
        stop = time.perf_counter()
        print(
            f"Query time + data transfer took: {stop - start} s, vespa said searchtime was {response.json.get('timing', {}).get('searchtime', -1)} s"
        )
    return format_query_results(query, response)


def float_to_binary_embedding(float_query_embedding: dict) -> dict:
    binary_query_embeddings = {}
    for k, v in float_query_embedding.items():
        binary_vector = (
            np.packbits(np.where(np.array(v) > 0, 1, 0)).astype(np.int8).tolist()
        )
        binary_query_embeddings[k] = binary_vector
        if len(binary_query_embeddings) >= MAX_QUERY_TERMS:
            print(f"Warning: Query has more than {MAX_QUERY_TERMS} terms. Truncating.")
            break
    return binary_query_embeddings


def create_nn_query_strings(
    binary_query_embeddings: dict, target_hits_per_query_tensor: int = 20
) -> Tuple[str, dict]:
    # Query tensors for nearest neighbor calculations
    nn_query_dict = {}
    for i in range(len(binary_query_embeddings)):
        nn_query_dict[f"input.query(rq{i})"] = binary_query_embeddings[i]
    nn = " OR ".join(
        [
            f"({{targetHits:{target_hits_per_query_tensor}}}nearestNeighbor(embedding,rq{i}))"
            for i in range(len(binary_query_embeddings))
        ]
    )
    return nn, nn_query_dict


def format_q_embs(q_embs: torch.Tensor) -> dict:
    float_query_embedding = {k: v.tolist() for k, v in enumerate(q_embs)}
    return float_query_embedding


async def get_result_from_query(
    app: Vespa,
    query: str,
    q_embs: torch.Tensor,
    ranking: str,
    token_to_idx: dict,
) -> Dict[str, Any]:
    # Get the query embeddings and token map
    print(query)

    print(token_to_idx)
    if ranking == "nn+colpali":
        result = await query_vespa_nearest_neighbor(app, query, q_embs)
    elif ranking == "bm25+colpali":
        result = await query_vespa_default(app, query, q_embs)
    elif ranking == "bm25":
        result = await query_vespa_bm25(app, query, q_embs)
    else:
        raise ValueError(f"Unsupported ranking: {ranking}")
    # Print score, title id, and text of the results
    for idx, child in enumerate(result["root"]["children"]):
        print(
            f"Result {idx+1}: {child['relevance']}, {child['fields']['title']}, {child['fields']['id']}"
        )
    for single_result in result["root"]["children"]:
        print(single_result["fields"].keys())
    return result


async def get_full_image_from_vespa(app: Vespa, id: str) -> str:
    async with app.asyncio(connections=1) as session:
        start = time.perf_counter()
        response: VespaQueryResponse = await session.query(
            body={
                "yql": f'select full_image from pdf_page where id contains "{id}"',
                "ranking": "unranked",
                "presentation.timing": True,
            },
        )
        assert response.is_successful(), response.json
        stop = time.perf_counter()
        print(
            f"Getting image from Vespa took: {stop - start} s, vespa said searchtime was {response.json.get('timing', {}).get('searchtime', -1)} s"
        )
    return response.json["root"]["children"][0]["fields"]["full_image"]


async def query_vespa_nearest_neighbor(
    app: Vespa,
    query: str,
    q_emb: torch.Tensor,
    target_hits_per_query_tensor: int = 20,
    hits: int = 3,
    timeout: str = "10s",
    **kwargs,
) -> dict:
    # Hyperparameter for speed vs. accuracy
    async with app.asyncio(connections=1) as session:
        float_query_embedding = format_q_embs(q_emb)
        binary_query_embeddings = float_to_binary_embedding(float_query_embedding)

        # Mixed tensors for MaxSim calculations
        query_tensors = {
            "input.query(qtb)": binary_query_embeddings,
            "input.query(qt)": float_query_embedding,
        }
        nn_string, nn_query_dict = create_nn_query_strings(
            binary_query_embeddings, target_hits_per_query_tensor
        )
        query_tensors.update(nn_query_dict)
        response: VespaQueryResponse = await session.query(
            body={
                **query_tensors,
                "presentation.timing": True,
                # if we use rank({nn_string}, userQuery()), dynamic summary doesn't work, see https://github.com/vespa-engine/vespa/issues/28704
                "yql": f"select id,title,snippet,text,url,blur_image,page_number,summaryfeatures from pdf_page where {nn_string} or userQuery()",
                "ranking.profile": "retrieval-and-rerank",
                "timeout": timeout,
                "hits": hits,
                "query": query,
                **kwargs,
            },
        )
        assert response.is_successful(), response.json
    return format_query_results(query, response)
