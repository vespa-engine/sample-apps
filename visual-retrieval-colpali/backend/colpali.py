#!/usr/bin/env python3

import torch
from PIL import Image
import numpy as np
from typing import cast

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from vespa.application import Vespa
from vespa.io import VespaQueryResponse

MAX_QUERY_TERMS = 64


def load_model():
    model_name = "vidore/colpali-v1.2"

    device = get_torch_device("auto")
    print(f"Using device: {device}")

    # Load the model
    model = cast(
        ColPali,
        ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        ),
    ).eval()

    # Load the processor
    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))
    return model, processor


# Create dummy image
dummy_image = Image.new("RGB", (448, 448), (255, 255, 255))


def process_queries(processor, queries, image):
    inputs = processor(
        images=[image] * len(queries), text=queries, return_tensors="pt", padding=True
    )
    return inputs


def display_query_results(query, response, hits=5):
    query_time = response.json.get("timing", {}).get("searchtime", -1)
    query_time = round(query_time, 2)
    count = response.json.get("root", {}).get("fields", {}).get("totalCount", 0)
    result_text = f"Query text: '{query}', query time {query_time}s, count={count}, top results:\n"
    print(result_text)
    return response.json


async def query_vespa_default(app: Vespa, query, q):
    async with app.asyncio(connections=1, total_timeout=120) as session:
        query_embedding = {k: v.tolist() for k, v in enumerate(q)}
        response: VespaQueryResponse = await session.query(
            body={
                "yql": "select id,title,url,image,page_number from pdf_page where userQuery();",
                "ranking": "default",
                "query": query,
                "timeout": "10s",
                "hits": 3,
                "input.query(qt)": query_embedding,
                "presentation.timing": True,
            },
        )
        assert response.is_successful(), response.json
    return display_query_results(query, response)


async def query_vespa_nearest_neighbor(app: Vespa, query, q):
    target_hits_per_query_tensor = 20  # Hyperparameter for speed vs. accuracy
    async with app.asyncio(connections=1, total_timeout=180) as session:
        float_query_embedding = {k: v.tolist() for k, v in enumerate(q)}
        binary_query_embeddings = {}
        for k, v in float_query_embedding.items():
            binary_vector = (
                np.packbits(np.where(np.array(v) > 0, 1, 0)).astype(np.int8).tolist()
            )
            binary_query_embeddings[k] = binary_vector
            if len(binary_query_embeddings) >= MAX_QUERY_TERMS:
                print(
                    f"Warning: Query has more than {MAX_QUERY_TERMS} terms. Truncating."
                )
                break

        # Mixed tensors for MaxSim calculations
        query_tensors = {
            "input.query(qtb)": binary_query_embeddings,
            "input.query(qt)": float_query_embedding,
        }

        # Query tensors for nearest neighbor calculations
        for i in range(len(binary_query_embeddings)):
            query_tensors[f"input.query(rq{i})"] = binary_query_embeddings[i]
        nn = " OR ".join(
            [
                f"({{targetHits:{target_hits_per_query_tensor}}}nearestNeighbor(embedding,rq{i}))"
                for i in range(len(binary_query_embeddings))
            ]
        )

        response: VespaQueryResponse = await session.query(
            body={
                **query_tensors,
                "presentation.timing": True,
                "yql": f"select id,title, url, image, page_number from pdf_page where {nn}",
                "ranking.profile": "retrieval-and-rerank",
                "timeout": "10s",
                "hits": 3,
            },
        )
        assert response.is_successful(), response.json
    return display_query_results(query, response)


async def get_result_from_query(
    app: Vespa, processor: ColPaliProcessor, model: ColPali, query: str, nn=False
):
    # Process the single query
    batch_query = process_queries(processor, [query], dummy_image)
    with torch.no_grad():
        batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
        embeddings_query = model(**batch_query)
        q = embeddings_query.to("cpu")[0]  # Extract the single embedding
    print(query)
    if nn:
        result = await query_vespa_nearest_neighbor(app, query, q)
    else:
        result = await query_vespa_default(app, query, q)
    return result


def get_result_dummy(query: str, nn: bool = False):
    result = {}
    result["timing"] = {}
    result["timing"]["querytime"] = 0.23700000000000002
    result["timing"]["summaryfetchtime"] = 0.001
    result["timing"]["searchtime"] = 0.23900000000000002
    result["root"] = {}
    result["root"]["id"] = "toplevel"
    result["root"]["relevance"] = 1
    result["root"]["fields"] = {}
    result["root"]["fields"]["totalCount"] = 59
    result["root"]["coverage"] = {}
    result["root"]["coverage"]["coverage"] = 100
    result["root"]["coverage"]["documents"] = 155
    result["root"]["coverage"]["full"] = True
    result["root"]["coverage"]["nodes"] = 1
    result["root"]["coverage"]["results"] = 1
    result["root"]["coverage"]["resultsFull"] = 1
    result["root"]["children"] = []
    elt0 = {}
    elt0["id"] = "index:colpalidemo_content/0/424c85e7dece761d226f060f"
    elt0["relevance"] = 2354.050122871995
    elt0["source"] = "colpalidemo_content"
    elt0["fields"] = {}
    elt0["fields"]["id"] = "a767cb1868be9a776cd56b768347b089"
    elt0["fields"]["url"] = (
        "https://static.conocophillips.com/files/resources/conocophillips-2023-sustainability-report.pdf"
    )
    elt0["fields"]["title"] = "ConocoPhillips 2023 Sustainability Report"
    elt0["fields"]["page_number"] = 50
    elt0["fields"]["image"] = "empty for now - is base64 encoded image"
    result["root"]["children"].append(elt0)
    elt1 = {}
    elt1["id"] = "index:colpalidemo_content/0/b927c4979f0beaf0d7fab8e9"
    elt1["relevance"] = 2313.7529950886965
    elt1["source"] = "colpalidemo_content"
    elt1["fields"] = {}
    elt1["fields"]["id"] = "9f2fc0aa02c9561adfaa1451c875658f"
    elt1["fields"]["url"] = (
        "https://static.conocophillips.com/files/resources/conocophillips-2023-managing-climate-related-risks.pdf"
    )
    elt1["fields"]["title"] = "ConocoPhillips Managing Climate Related Risks"
    elt1["fields"]["page_number"] = 44
    elt1["fields"]["image"] = "empty for now - is base64 encoded image"
    result["root"]["children"].append(elt1)
    elt2 = {}
    elt2["id"] = "index:colpalidemo_content/0/9632d72238829d6afefba6c9"
    elt2["relevance"] = 2312.230182081461
    elt2["source"] = "colpalidemo_content"
    elt2["fields"] = {}
    elt2["fields"]["id"] = "d638ded1ddcb446268b289b3f65430fd"
    elt2["fields"]["url"] = (
        "https://static.conocophillips.com/files/resources/24-0976-sustainability-highlights_nature.pdf"
    )
    elt2["fields"]["title"] = (
        "ConocoPhillips Sustainability Highlights - Nature (24-0976)"
    )
    elt2["fields"]["page_number"] = 0
    elt2["fields"]["image"] = "empty for now - is base64 encoded image"
    result["root"]["children"].append(elt2)
    return result
