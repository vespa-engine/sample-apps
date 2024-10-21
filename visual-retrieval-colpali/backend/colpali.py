#!/usr/bin/env python3

import torch
from PIL import Image
import numpy as np
from typing import Generator
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

from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.interpretability import (
    normalize_similarity_map,
    get_similarity_maps_from_embeddings,
)
from colpali_engine.utils.torch_utils import get_torch_device
from einops import rearrange
from vidore_benchmark.interpretability.vit_configs import VIT_CONFIG
from vespa.application import Vespa
from vespa.io import VespaQueryResponse

matplotlib.use("Agg")

MAX_QUERY_TERMS = 64

MODEL_NAME = "vidore/colqwen2-v0.1"
N_PATCHES_PER_DIM = 14


def load_model() -> Tuple[ColQwen2, ColQwen2Processor]:
    device = get_torch_device("auto")
    print(f"Using device: {device}")

    # Load the model
    model = ColQwen2.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    ).eval()
    # Load the processor
    processor = ColQwen2Processor.from_pretrained(MODEL_NAME)
    return model, processor


def load_vit_config(model):
    # Load the ViT config
    print(f"VIT config: {VIT_CONFIG}")
    vit_config = None  # VIT_CONFIG[ColQwen2_GEMMA_MODEL_NAME]
    return None


def gen_similarity_maps(
    model: ColQwen2,
    processor: ColQwen2Processor,
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
        model (ColQwen2): The model used for generating embeddings.
        processor (ColQwen2Processor): Processor for images and text.
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
    n_patches = processor.get_n_patches(
        original_sizes[0], model.patch_size, model.spatial_merge_size
    )
    n_x, n_y = n_patches
    max_patch = n_x * n_y
    batch_images = processor.process_images(original_images).to(device)
    image_mask = processor.get_image_mask(batch_images)
    image_mask_npy = image_mask.cpu().numpy().squeeze()
    # Mask is false for idx 0:4 and -7: - In total 11 patches are masked
    # 746-11 = 735 (0-indexed) -> 23x32 = 736 patches
    print(f" N patches : {n_patches}")
    print(f" Image mask: {image_mask_npy.shape}")
    # If similarity maps are provided, use them instead of computing them
    if True:
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
                len(image_mask_npy[0]),
            )
        )
        # N patches : (23, 32)
        # Using provided similarity maps
        # Max qt: 26, max patch: 746
        for idx, vespa_sim_map in enumerate(vespa_sim_maps):
            for cell in vespa_sim_map["similarities"]["cells"]:
                patch = int(cell["address"]["patch"])
                query_token = int(cell["address"]["querytoken"])
                value = cell["value"]
                vespa_sim_map_tensor[idx, query_token, patch] = value
        # apply image mask
        vespa_sim_map_tensor = vespa_sim_map_tensor[:, :, image_mask_npy[0]]
        vespa_sim_map_tensor = rearrange(
            vespa_sim_map_tensor, "i qt (x y) -> i qt x y", x=n_x, y=n_y
        )
        for idx in range(len(vespa_sim_map_tensor)):
            vespa_sim_map_tensor[idx] = normalize_similarity_map(
                vespa_sim_map_tensor[idx]
            )
        similarity_map_normalized = vespa_sim_map_tensor  # normalize_similarity_map_per_query_token(vespa_sim_map_tensor)
    else:
        # Preprocess inputs
        print("Computing similarity maps")
        start2 = time.perf_counter()
        input_image_processed = processor.process_images(processed_images).to(device)

        # Forward passes
        with torch.no_grad():
            output_image = model.forward(**input_image_processed)
        # query_embs are (n_tokens, hidden_size)
        # need to repeat for each image (len(images), n_tokens, hidden_size)
        q_embs = query_embs.unsqueeze(0).repeat(len(images), 1, 1).to(device)
        # Print shape of
        print(f"Query embeddings shape: {q_embs.shape}")
        print(f"Output image shape: {output_image.shape}")
        print(f"N patches: {n_patches}")
        batched_similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=output_image,
            query_embeddings=q_embs,
            n_patches=n_patches,
            image_mask=image_mask,
        )
        # normalize similarity maps
        similarity_map_normalized = []
        for idx in range(len(batched_similarity_maps)):
            similarity_map_normalized.append(
                normalize_similarity_map(batched_similarity_maps[idx])
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
            print(f"Creating similarity map for token: {token}")
            print(f"Token index: {token_idx}")
            # Get the similarity map for this image and the selected token
            if isinstance(similarity_map_normalized, list):
                sim_map = similarity_map_normalized[idx][
                    token_idx, :, :
                ]  # Shape: (h, w)
            elif isinstance(similarity_map_normalized, torch.Tensor):
                sim_map = similarity_map_normalized[idx, token_idx, :, :]
            # Reshape the similarity map to match the PIL shape convention
            sim_map = rearrange(sim_map, "h w -> w h")
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


def replace_tokens(tokens: str) -> str:
    """Replace Ġ with _"""
    return tokens.replace("Ġ", "_")


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
    processor: ColQwen2Processor,
    model: ColQwen2,
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
    model: ColQwen2,
    processor: ColQwen2Processor,
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
            # save_figure(fig, f"similarity_map_{token}.png")
    print("Done")
