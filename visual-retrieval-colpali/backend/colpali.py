#!/usr/bin/env python3

import torch
from PIL import Image
import numpy as np
from typing import cast, Generator
from pathlib import Path
import base64
from io import BytesIO
from typing import Union, Tuple, List
import matplotlib
import matplotlib.cm as cm
import re
import io

import time
import backend.testquery as testquery

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from einops import rearrange
from vidore_benchmark.interpretability.torch_utils import (
    normalize_similarity_map_per_query_token,
)
from vidore_benchmark.interpretability.vit_configs import VIT_CONFIG

matplotlib.use("Agg")
# Prepare the colormap once to avoid recomputation
colormap = cm.get_cmap("viridis")

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
    return model, processor, device


def load_vit_config(model):
    # Load the ViT config
    print(f"VIT config: {VIT_CONFIG}")
    vit_config = VIT_CONFIG[COLPALI_GEMMA_MODEL_NAME]
    return vit_config


def gen_similarity_maps(
    model: ColPali,
    processor: ColPaliProcessor,
    device,
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
        token_idx_map (dict): Mapping from indices to tokens.
        images (List[Union[Path, str]]): List of image paths or base64-encoded strings.
        vespa_sim_maps (List[str]): List of Vespa similarity maps.

    Yields:
        Tuple[int, str, str]: A tuple containing the image index, the selected token, and the base64-encoded image.

    """
    vit_config = load_vit_config(model)
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
        # "quantized": [
        #      {
        #        "address": {
        #          "patch": "0",
        #          "querytoken": "0"
        #        },
        #        "value": 12, # score in range [-128, 127]
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
            for cell in vespa_sim_map["quantized"]["cells"]:
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
        for token_idx, token in token_idx_map.items():
            if should_filter_token(token):
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
            yield idx, token, token_idx, blended_img_base64
    end3 = time.perf_counter()
    print(f"Blending images took: {end3 - start3} s")


def get_query_embeddings_and_token_map(
    processor, model, query
) -> Tuple[torch.Tensor, dict]:
    if model is None:  # use static test query data (saves time when testing)
        return testquery.q_embs, testquery.idx_to_token

    start_time = time.perf_counter()
    inputs = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        embeddings_query = model(**inputs)
        q_emb = embeddings_query.to("cpu")[0]  # Extract the single embedding
    # Use this cell output to choose a token using its index
    query_tokens = processor.tokenizer.tokenize(processor.decode(inputs.input_ids[0]))
    # reverse key, values in dictionary
    print(query_tokens)
    idx_to_token = {idx: val for idx, val in enumerate(query_tokens)}
    end_time = time.perf_counter()
    print(f"Query inference took: {end_time - start_time} s")
    return q_emb, idx_to_token


def should_filter_token(token: str) -> bool:
    # Pattern to match tokens that start with '<', numbers, whitespace, special characters (except ▁), or the string 'Question'
    # Will exclude these tokens from the similarity map generation
    # Does NOT match:
    # 2
    # 0
    # 2
    # 3
    # ▁2
    # ▁hi
    #
    # Do match:
    # <bos>
    # Question
    # :
    # _Percentage
    # <pad>
    # \n
    # ▁
    # ?
    # )
    # %
    # /)
    pattern = re.compile(r"^<.*$|^\s+$|^(?!.*\d)(?!▁)\S+$|^Question$|^▁$")
    if pattern.match(token):
        return True
    return False
