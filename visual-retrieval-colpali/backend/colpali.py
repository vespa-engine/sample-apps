#!/usr/bin/env python3

import torch
from PIL import Image
import numpy as np
from typing import cast
import pprint
from pathlib import Path

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from einops import rearrange
from vidore_benchmark.interpretability.plot_utils import plot_similarity_heatmap
from vidore_benchmark.interpretability.torch_utils import (
    normalize_similarity_map_per_query_token,
)
from vidore_benchmark.interpretability.vit_configs import VIT_CONFIG
from vidore_benchmark.utils.image_utils import scale_image
from vespa.application import Vespa
from vespa.io import VespaQueryResponse

MAX_QUERY_TERMS = 64
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "sim_maps"
OUTPUT_DIR.mkdir(exist_ok=True)


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


def load_vit_config(model):
    # Load the ViT config
    vit_config = VIT_CONFIG[model.config.name_or_path]
    return vit_config


# Create dummy image
dummy_image = Image.new("RGB", (448, 448), (255, 255, 255))


def get_token_dict_from_query(processor, device, query):
    # We should refactor to only do the query inference once
    # But we need the map of tokens and their indices to choose the token
    # Preprocess inputs
    input_text_processed = processor.process_queries([query]).to(device)
    with torch.no_grad():
        output_text = model.forward(**input_text_processed)
    # Use this cell output to choose a token using its index
    query_tokens = processor.tokenizer.tokenize(
        processor.decode(input_text_processed.input_ids[0])
    )
    # reverse key, values in dictionary
    return {val: idx for idx, val in query_tokens.items()}


def gen_similarity_map(model, processor, device, vit_config, query, image=None):
    # Should take in the b64 image from Vespa query result
    # And possibly the tensor representing the output_image
    if image is None:
        # Use the default image
        image_filepath = (
            Path(__file__).parent.parent
            / "static"
            / "assets"
            / "ConocoPhillips Sustainability Highlights - Nature (24-0976).png"
        )
        query = "How many percent of source water is fresh water?"
        image = Image.open(image_filepath)
    else:
        # image is b64 image
        image = Image.open(image)
    # Preview the image
    scale_image(image, 512)
    # Preprocess inputs
    input_text_processed = processor.process_queries([query]).to(device)
    input_image_processed = processor.process_images([image]).to(device)
    # Forward passes
    with torch.no_grad():
        output_text = model.forward(**input_text_processed)
        output_image = model.forward(**input_image_processed)
    # output_image is the tensor that we could get from the Vespa query
    # Print shape of output_text and output_image
    # Output image shape: torch.Size([1, 1030, 128])
    # Remove the special tokens from the output
    output_image = output_image[
        :, : processor.image_seq_length, :
    ]  # (1, n_patches_x * n_patches_y, dim)

    # Rearrange the output image tensor to explicitly represent the 2D grid of patches
    output_image = rearrange(
        output_image,
        "b (h w) c -> b h w c",
        h=vit_config.n_patch_per_dim,
        w=vit_config.n_patch_per_dim,
    )  # (1, n_patches_x, n_patches_y, dim)
    # Get the similarity map
    similarity_map = torch.einsum(
        "bnk,bijk->bnij", output_text, output_image
    )  # (1, query_tokens, n_patches_x, n_patches_y)

    # Normalize the similarity map
    similarity_map_normalized = normalize_similarity_map_per_query_token(
        similarity_map
    )  # (1, query_tokens, n_patches_x, n_patches_y)
    # Use this cell output to choose a token using its index
    query_tokens = processor.tokenizer.tokenize(
        processor.decode(input_text_processed.input_ids[0])
    )
    # Choose a token
    token_idx = (
        10  # e.g. if "12: '▁Kazakhstan',", set 12 to choose the token 'Kazakhstan'
    )
    selected_token = processor.decode(input_text_processed.input_ids[0, token_idx])
    # strip whitespace
    selected_token = selected_token.strip()
    print(f"Selected token: `{selected_token}`")
    # Retrieve the similarity map for the chosen token
    pprint.pprint({idx: val for idx, val in enumerate(query_tokens)})
    # Resize the image to square
    input_image_square = image.resize((vit_config.resolution, vit_config.resolution))

    # Plot the similarity map
    fig, ax = plot_similarity_heatmap(
        input_image_square,
        patch_size=vit_config.patch_size,
        image_resolution=vit_config.resolution,
        similarity_map=similarity_map_normalized[0, token_idx, :, :],
    )
    # Add the query text
    ax.set_title(query, fontsize=18)
    # Add annotation with selected token
    ax.annotate(
        f"Selected token:`{selected_token}`",
        xy=(0.5, 0.95),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=18,
        color="black",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
    )

    fig.savefig(
        OUTPUT_DIR / "similarity_map.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    return similarity_map_normalized


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


if __name__ == "__main__":
    model, processor = load_model()
    vit_config = load_vit_config(model)
    query = "Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?"
    gen_similarity_map(model, processor, model.device, vit_config, query=query)
    result = get_result_dummy("dummy query")
    print(result)
    print("Done")
