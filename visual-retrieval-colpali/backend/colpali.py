#!/usr/bin/env python3

import torch
from PIL import Image
import numpy as np
from typing import cast, Generator
from pathlib import Path
import base64
from io import BytesIO
from typing import Union, Tuple, List, Dict, Any
import matplotlib
import matplotlib.cm as cm
import re
import io

import json
import time
import backend.testquery as testquery

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from einops import rearrange
from vidore_benchmark.interpretability.torch_utils import (
    normalize_similarity_map_per_query_token,
)
from vidore_benchmark.interpretability.vit_configs import VIT_CONFIG
from vespa.application import Vespa
from vespa.io import VespaQueryResponse

matplotlib.use("Agg")

MAX_QUERY_TERMS = 64

COLPALI_GEMMA_MODEL_NAME = "vidore/colpaligemma-3b-pt-448-base"


def load_model() -> Tuple[ColPali, ColPaliProcessor]:
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
    print(f"VIT config: {VIT_CONFIG}")
    vit_config = VIT_CONFIG[COLPALI_GEMMA_MODEL_NAME]
    return vit_config


def save_figure(fig, filename: str = "similarity_map.png"):
    try:
        OUTPUT_DIR = Path(__file__).parent.parent / "output" / "sim_maps"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            OUTPUT_DIR / filename,
            bbox_inches="tight",
            pad_inches=0,
        )
    except Exception as e:
        print(f"Failed to save figure: {e}")


def annotate_plot(ax, query, selected_token):
    # Add the query text as a title over the image with opacity
    ax.text(
        0.5,
        0.95,  # Adjust the position to be on the image (y=0.1 is 10% from the bottom)
        query,
        fontsize=18,
        color="white",
        ha="center",
        va="center",
        alpha=0.8,  # Set opacity (1 is fully opaque, 0 is fully transparent)
        bbox=dict(
            boxstyle="round,pad=0.5", fc="black", ec="none", lw=0, alpha=0.5
        ),  # Add a semi-transparent background
        transform=ax.transAxes,  # Ensure the coordinates are relative to the axes
    )

    # Add annotation with the selected token over the image with opacity
    ax.text(
        0.5,
        0.05,  # Position towards the top of the image
        f"Selected token: `{selected_token}`",
        fontsize=18,
        color="white",
        ha="center",
        va="center",
        alpha=0.8,  # Set opacity for the text
        bbox=dict(
            boxstyle="round,pad=0.3", fc="black", ec="none", lw=0, alpha=0.5
        ),  # Semi-transparent background
        transform=ax.transAxes,  # Keep the coordinates relative to the axes
    )
    return ax


def gen_similarity_maps(
    model: ColPali,
    processor: ColPaliProcessor,
    device,
    vit_config,
    query: str,
    query_embs: torch.Tensor,
    token_idx_map: dict,
    images: List[Union[Path, str]],
    vespa_sim_maps: List[str],
) -> Generator[Tuple[int, str, str], None, None]:
    """
    Generate similarity maps for the given images and query, and return base64-encoded blended images.

    Args:
        model (ColPali): The model used for generating embeddings.
        processor (ColPaliProcessor): Processor for images and text.
        device: Device to run the computations on.
        vit_config: Configuration for the Vision Transformer.
        query (str): The query string.
        query_embs (torch.Tensor): Query embeddings.
        token_idx_map (dict): Mapping from tokens to their indices.
        images (List[Union[Path, str]]): List of image paths or base64-encoded strings.
        vespa_sim_maps (List[str]): List of Vespa similarity maps.

    Yields:
        Tuple[int, str, str]: A tuple containing the image index, the selected token, and the base64-encoded image.

    """

    # Prepare the colormap once to avoid recomputation
    colormap = cm.get_cmap("viridis")

    # Process images and store original images and sizes
    processed_images = []
    original_images = []
    original_sizes = []
    for img in images:
        if isinstance(img, Path):
            try:
                img_pil = Image.open(img).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to open image from path: {e}")
        elif isinstance(img, str):
            try:
                img_pil = Image.open(BytesIO(base64.b64decode(img))).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to open image from base64 string: {e}")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        original_images.append(img_pil.copy())
        original_sizes.append(img_pil.size)  # (width, height)
        processed_images.append(img_pil)

    # If similarity maps are provided, use them instead of computing them
    if vespa_sim_maps:
        print("Using provided similarity maps")
        # A sim map looks like this:
        # "similarities": [
        #      {
        #        "address": {
        #          "patch": "0",
        #          "querytoken": "0"
        #        },
        #        "value": 1.2599412202835083
        #      },
        # ... and so on.
        # Now turn these into a tensor of same shape as previous similarity map
        vespa_sim_map_tensor = torch.zeros(
            (
                len(vespa_sim_maps),
                query_embs.size(dim=1),
                vit_config.n_patch_per_dim,
                vit_config.n_patch_per_dim,
            )
        )
        for idx, vespa_sim_map in enumerate(vespa_sim_maps):
            for cell in vespa_sim_map["similarities"]["cells"]:
                patch = int(cell["address"]["patch"])
                # if dummy model then just use 1024 as the image_seq_length

                if hasattr(processor, "image_seq_length"):
                    image_seq_length = processor.image_seq_length
                else:
                    image_seq_length = 1024

                if patch >= image_seq_length:
                    continue
                query_token = int(cell["address"]["querytoken"])
                value = cell["value"]
                vespa_sim_map_tensor[
                    idx,
                    int(query_token),
                    int(patch) // vit_config.n_patch_per_dim,
                    int(patch) % vit_config.n_patch_per_dim,
                ] = value

        # Normalize the similarity map per query token
        similarity_map_normalized = normalize_similarity_map_per_query_token(
            vespa_sim_map_tensor
        )
    else:
        # Preprocess inputs
        print("Computing similarity maps")
        start2 = time.perf_counter()
        input_image_processed = processor.process_images(processed_images).to(device)

        # Forward passes
        with torch.no_grad():
            output_image = model.forward(**input_image_processed)

        # Remove the special tokens from the output
        output_image = output_image[:, : processor.image_seq_length, :]

        # Rearrange the output image tensor to represent the 2D grid of patches
        output_image = rearrange(
            output_image,
            "b (h w) c -> b h w c",
            h=vit_config.n_patch_per_dim,
            w=vit_config.n_patch_per_dim,
        )

        # Ensure query_embs has batch dimension
        if query_embs.dim() == 2:
            query_embs = query_embs.unsqueeze(0).to(device)
        else:
            query_embs = query_embs.to(device)

        # Compute the similarity map
        similarity_map = torch.einsum(
            "bnk,bhwk->bnhw", query_embs, output_image
        )  # Shape: (batch_size, query_tokens, h, w)

        end2 = time.perf_counter()
        print(f"Similarity map computation took: {end2 - start2} s")

        # Normalize the similarity map per query token
        similarity_map_normalized = normalize_similarity_map_per_query_token(
            similarity_map
        )

    # Collect the blended images
    start3 = time.perf_counter()
    for idx, img in enumerate(original_images):
        SCALING_FACTOR = 8
        sim_map_resolution = (
            max(32, int(original_sizes[idx][0] / SCALING_FACTOR)),
            max(32, int(original_sizes[idx][1] / SCALING_FACTOR)),
        )

        result_per_image = {}
        for token, token_idx in token_idx_map.items():
            if is_special_token(token):
                continue

            # Get the similarity map for this image and the selected token
            sim_map = similarity_map_normalized[idx, token_idx, :, :]  # Shape: (h, w)

            # Move the similarity map to CPU, convert to float (as BFloat16 not supported by Numpy) and convert to NumPy array
            sim_map_np = sim_map.cpu().float().numpy()

            # Resize the similarity map to the original image size
            sim_map_img = Image.fromarray(sim_map_np)
            sim_map_resized = sim_map_img.resize(
                sim_map_resolution, resample=Image.BICUBIC
            )

            # Convert the resized similarity map to a NumPy array
            sim_map_resized_np = np.array(sim_map_resized, dtype=np.float32)

            # Normalize the similarity map to range [0, 1]
            sim_map_min = sim_map_resized_np.min()
            sim_map_max = sim_map_resized_np.max()
            if sim_map_max - sim_map_min > 1e-6:
                sim_map_normalized = (sim_map_resized_np - sim_map_min) / (
                    sim_map_max - sim_map_min
                )
            else:
                sim_map_normalized = np.zeros_like(sim_map_resized_np)

            # Apply a colormap to the normalized similarity map
            heatmap = colormap(sim_map_normalized)  # Returns an RGBA array

            # Convert the heatmap to a PIL Image
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            heatmap_img = Image.fromarray(heatmap_uint8)
            heatmap_img_rgba = heatmap_img.convert("RGBA")

            # Save the image to a BytesIO buffer
            buffer = io.BytesIO()
            heatmap_img_rgba.save(buffer, format="PNG")
            buffer.seek(0)

            # Encode the image to base64
            blended_img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

            # Store the base64-encoded image
            result_per_image[token] = blended_img_base64
            yield idx, token, blended_img_base64
    end3 = time.perf_counter()
    print(f"Blending images took: {end3 - start3} s")


def get_query_embeddings_and_token_map(
    processor, model, query
) -> Tuple[torch.Tensor, dict]:
    if model is None:  # use static test query data (saves time when testing)
        return testquery.q_embs, testquery.token_to_idx

    start_time = time.perf_counter()
    inputs = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        embeddings_query = model(**inputs)
        q_emb = embeddings_query.to("cpu")[0]  # Extract the single embedding
    # Use this cell output to choose a token using its index
    query_tokens = processor.tokenizer.tokenize(processor.decode(inputs.input_ids[0]))
    # reverse key, values in dictionary
    print(query_tokens)
    token_to_idx = {val: idx for idx, val in enumerate(query_tokens)}
    end_time = time.perf_counter()
    print(f"Query inference took: {end_time - start_time} s")
    return q_emb, token_to_idx


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
    async with app.asyncio(connections=1, total_timeout=120) as session:
        query_embedding = format_q_embs(q_emb)

        start = time.perf_counter()
        response: VespaQueryResponse = await session.query(
            body={
                "yql": "select id,title,url,full_image,page_number,snippet,text,summaryfeatures from pdf_page where userQuery();",
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
    async with app.asyncio(connections=1, total_timeout=120) as session:
        query_embedding = format_q_embs(q_emb)

        start = time.perf_counter()
        response: VespaQueryResponse = await session.query(
            body={
                "yql": "select id,title,url,full_image,page_number,snippet,text,summaryfeatures from pdf_page where userQuery();",
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
    async with app.asyncio(connections=1, total_timeout=180) as session:
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
                "yql": f"select id,title,snippet,text,url,full_image,page_number,summaryfeatures from pdf_page where {nn_string} or userQuery()",
                "ranking.profile": "retrieval-and-rerank",
                "timeout": timeout,
                "hits": hits,
                "query": query,
                **kwargs,
            },
        )
        assert response.is_successful(), response.json
    return format_query_results(query, response)


def is_special_token(token: str) -> bool:
    # Pattern for tokens that start with '<', numbers, whitespace, or single characters, or the string 'Question'
    # Will exclude these tokens from the similarity map generation
    pattern = re.compile(r"^<.*$|^\d+$|^\s+$|^\w$|^Question$")
    if (len(token) < 3) or pattern.match(token):
        return True
    return False


async def get_result_from_query(
    app: Vespa,
    processor: ColPaliProcessor,
    model: ColPali,
    query: str,
    q_embs: torch.Tensor,
    token_to_idx: Dict[str, int],
    ranking: str,
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


def add_sim_maps_to_result(
    result: Dict[str, Any],
    model: ColPali,
    processor: ColPaliProcessor,
    query: str,
    q_embs: Any,
    token_to_idx: Dict[str, int],
    query_id: str,
    result_cache,
) -> Dict[str, Any]:
    vit_config = load_vit_config(model)
    imgs: List[str] = []
    vespa_sim_maps: List[str] = []
    for single_result in result["root"]["children"]:
        img = single_result["fields"]["full_image"]
        if img:
            imgs.append(img)
        vespa_sim_map = single_result["fields"].get("summaryfeatures", None)
        if vespa_sim_map:
            vespa_sim_maps.append(vespa_sim_map)
    sim_map_imgs_generator = gen_similarity_maps(
        model=model,
        processor=processor,
        device=model.device if hasattr(model, "device") else "cpu",
        vit_config=vit_config,
        query=query,
        query_embs=q_embs,
        token_idx_map=token_to_idx,
        images=imgs,
        vespa_sim_maps=vespa_sim_maps,
    )
    for img_idx, token, sim_mapb64 in sim_map_imgs_generator:
        print(f"Created sim map for image {img_idx} and token {token}")
        result["root"]["children"][img_idx]["fields"][f"sim_map_{token}"] = sim_mapb64
        # Update result_cache with the new sim_map
        result_cache.set(query_id, result)
    # for single_result, sim_map_dict in zip(result["root"]["children"], sim_map_imgs):
    #     for token, sim_mapb64 in sim_map_dict.items():
    #         single_result["fields"][f"sim_map_{token}"] = sim_mapb64
    return result


if __name__ == "__main__":
    model, processor = load_model()
    vit_config = load_vit_config(model)
    query = "How many percent of source water is fresh water?"
    image_filepath = (
        Path(__file__).parent.parent
        / "static"
        / "assets"
        / "ConocoPhillips Sustainability Highlights - Nature (24-0976).png"
    )
    q_embs, token_to_idx = get_query_embeddings_and_token_map(
        processor,
        model,
        query,
    )
    figs_images = gen_similarity_maps(
        model,
        processor,
        model.device,
        vit_config,
        query=query,
        query_embs=q_embs,
        token_idx_map=token_to_idx,
        images=[image_filepath],
        vespa_sim_maps=None,
    )
    for fig_token in figs_images:
        for token, (fig, ax) in fig_token.items():
            print(f"Token: {token}")
            save_figure(fig, f"similarity_map_{token}.png")
    print("Done")
