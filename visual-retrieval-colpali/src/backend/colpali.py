import torch
from PIL import Image
import numpy as np
from typing import Generator, Tuple, List, Union, Dict
from pathlib import Path
import base64
from io import BytesIO
import re
import io
import matplotlib.cm as cm

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from vidore_benchmark.interpretability.torch_utils import (
    normalize_similarity_map_per_query_token,
)
from functools import lru_cache
import logging


class SimMapGenerator:
    """
    Generates similarity maps based on query embeddings and image patches using the ColPali model.
    """

    colormap = cm.get_cmap("viridis")  # Preload colormap for efficiency

    def __init__(
        self,
        logger: logging.Logger,
        model_name: str = "vidore/colpali-v1.2",
        n_patch: int = 32,
    ):
        """
        Initializes the SimMapGenerator class with a specified model and patch dimension.

        Args:
            model_name (str): The model name for loading the ColPali model.
            n_patch (int): The number of patches per dimension.
        """
        self.model_name = model_name
        self.n_patch = n_patch
        self.device = get_torch_device("auto")
        self.logger = logger
        self.logger.info(f"Using device: {self.device}")
        self.model, self.processor = self.load_model()

    def load_model(self) -> Tuple[ColPali, ColPaliProcessor]:
        """
        Loads the ColPali model and processor.

        Returns:
            Tuple[ColPali, ColPaliProcessor]: Loaded model and processor.
        """
        model = ColPali.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,  # Note that the embeddings created during feed were float32 -> binarized, yet setting this seem to produce the most similar results both locally (mps) and HF (Cuda)
            device_map=self.device,
        ).eval()

        processor = ColPaliProcessor.from_pretrained(self.model_name)
        return model, processor

    def gen_similarity_maps(
        self,
        query: str,
        query_embs: torch.Tensor,
        token_idx_map: Dict[int, str],
        images: List[Union[Path, str]],
        vespa_sim_maps: List[Dict],
    ) -> Generator[Tuple[int, str, str], None, None]:
        """
        Generates similarity maps for the provided images and query, and returns base64-encoded blended images.

        Args:
            query (str): The query string.
            query_embs (torch.Tensor): Query embeddings tensor.
            token_idx_map (dict): Mapping from indices to tokens.
            images (List[Union[Path, str]]): List of image paths or base64-encoded strings.
            vespa_sim_maps (List[Dict]): List of Vespa similarity maps.

        Yields:
            Tuple[int, str, str]: A tuple containing the image index, selected token, and base64-encoded image.
        """
        processed_images, original_images, original_sizes = [], [], []
        for img in images:
            img_pil = self._load_image(img)
            original_images.append(img_pil.copy())
            original_sizes.append(img_pil.size)
            processed_images.append(img_pil)

        vespa_sim_map_tensor = self._prepare_similarity_map_tensor(
            query_embs, vespa_sim_maps
        )
        similarity_map_normalized = normalize_similarity_map_per_query_token(
            vespa_sim_map_tensor
        )

        for idx, img in enumerate(original_images):
            for token_idx, token in token_idx_map.items():
                if self.should_filter_token(token):
                    continue

                sim_map = similarity_map_normalized[idx, token_idx, :, :]
                blended_img_base64 = self._blend_image(
                    img, sim_map, original_sizes[idx]
                )
                yield idx, token, token_idx, blended_img_base64

    def _load_image(self, img: Union[Path, str]) -> Image:
        """
        Loads an image from a file path or a base64-encoded string.

        Args:
            img (Union[Path, str]): The image to load.

        Returns:
            Image: The loaded PIL image.
        """
        try:
            if isinstance(img, Path):
                return Image.open(img).convert("RGB")
            elif isinstance(img, str):
                return Image.open(BytesIO(base64.b64decode(img))).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

    def _prepare_similarity_map_tensor(
        self, query_embs: torch.Tensor, vespa_sim_maps: List[Dict]
    ) -> torch.Tensor:
        """
        Prepares a similarity map tensor from Vespa similarity maps.

        Args:
            query_embs (torch.Tensor): Query embeddings tensor.
            vespa_sim_maps (List[Dict]): List of Vespa similarity maps.

        Returns:
            torch.Tensor: The prepared similarity map tensor.
        """
        vespa_sim_map_tensor = torch.zeros(
            (len(vespa_sim_maps), query_embs.size(1), self.n_patch, self.n_patch)
        )
        for idx, vespa_sim_map in enumerate(vespa_sim_maps):
            for cell in vespa_sim_map["quantized"]["cells"]:
                patch = int(cell["address"]["patch"])
                query_token = int(cell["address"]["querytoken"])
                value = cell["value"]
                if hasattr(self.processor, "image_seq_length"):
                    image_seq_length = self.processor.image_seq_length
                else:
                    image_seq_length = 1024

                if patch >= image_seq_length:
                    continue
                vespa_sim_map_tensor[
                    idx,
                    query_token,
                    patch // self.n_patch,
                    patch % self.n_patch,
                ] = value
        return vespa_sim_map_tensor

    def _blend_image(
        self, img: Image, sim_map: torch.Tensor, original_size: Tuple[int, int]
    ) -> str:
        """
        Blends an image with a similarity map and encodes it to base64.

        Args:
            img (Image): The original image.
            sim_map (torch.Tensor): The similarity map tensor.
            original_size (Tuple[int, int]): The original size of the image.

        Returns:
            str: The base64-encoded blended image.
        """
        SCALING_FACTOR = 8
        sim_map_resolution = (
            max(32, int(original_size[0] / SCALING_FACTOR)),
            max(32, int(original_size[1] / SCALING_FACTOR)),
        )

        sim_map_np = sim_map.cpu().float().numpy()
        sim_map_img = Image.fromarray(sim_map_np).resize(
            sim_map_resolution, resample=Image.BICUBIC
        )
        sim_map_resized_np = np.array(sim_map_img, dtype=np.float32)
        sim_map_normalized = self._normalize_sim_map(sim_map_resized_np)

        heatmap = self.colormap(sim_map_normalized)
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).convert("RGBA")

        buffer = io.BytesIO()
        heatmap_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _normalize_sim_map(sim_map: np.ndarray) -> np.ndarray:
        """
        Normalizes a similarity map to range [0, 1].

        Args:
            sim_map (np.ndarray): The similarity map.

        Returns:
            np.ndarray: The normalized similarity map.
        """
        sim_map_min, sim_map_max = sim_map.min(), sim_map.max()
        if sim_map_max - sim_map_min > 1e-6:
            return (sim_map - sim_map_min) / (sim_map_max - sim_map_min)
        return np.zeros_like(sim_map)

    @staticmethod
    def should_filter_token(token: str) -> bool:
        """
        Determines if a token should be filtered out based on predefined patterns.

        The function filters out tokens that:

            - Start with '<' (e.g., '<bos>')
            - Consist entirely of whitespace
            - Are purely punctuation (excluding tokens that contain digits or start with '▁')
            - Start with an underscore '_'
            - Exactly match the word 'Question'
            - Are exactly the single character '▁'

        Output of test:
            Token: '2'         | False
            Token: '0'         | False
            Token: '2'         | False
            Token: '3'         | False
            Token: '▁2'        | False
            Token: '▁hi'       | False
            Token: 'norwegian' | False
            Token: 'unlisted'  | False
            Token: '<bos>'     | True
            Token: 'Question'  | True
            Token: ':'         | True
            Token: '<pad>'     | True
            Token: '\n'        | True
            Token: '▁'         | True
            Token: '?'         | True
            Token: ')'         | True
            Token: '%'         | True
            Token: '/)'        | True


        Args:
            token (str): The token to check.

        Returns:
            bool: True if the token should be filtered out, False otherwise.
        """
        pattern = re.compile(
            r"^<.*$|^\s+$|^(?!.*\d)(?!▁)[^\w\s]+$|^_.*$|^Question$|^▁$"
        )
        return bool(pattern.match(token))

    @lru_cache(maxsize=128)
    def get_query_embeddings_and_token_map(
        self, query: str
    ) -> Tuple[torch.Tensor, dict]:
        """
        Retrieves query embeddings and a token index map.

        Args:
            query (str): The query string.

        Returns:
            Tuple[torch.Tensor, dict]: Query embeddings and token index map.
        """
        inputs = self.processor.process_queries([query]).to(self.model.device)
        with torch.no_grad():
            q_emb = self.model(**inputs).to("cpu")[0]

        query_tokens = self.processor.tokenizer.tokenize(
            self.processor.decode(inputs.input_ids[0])
        )
        idx_to_token = {idx: token for idx, token in enumerate(query_tokens)}
        return q_emb, idx_to_token
