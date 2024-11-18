import os
import time
from typing import Any, Dict, Tuple
import asyncio
import numpy as np
import torch
from dotenv import load_dotenv
from vespa.application import Vespa
from vespa.io import VespaQueryResponse
from .colpali import SimMapGenerator
import backend.stopwords
import logging


class VespaQueryClient:
    MAX_QUERY_TERMS = 64
    VESPA_SCHEMA_NAME = "pdf_page"
    SELECT_FIELDS = "id,title,url,blur_image,page_number,snippet,text"

    def __init__(self, logger: logging.Logger):
        """
        Initialize the VespaQueryClient by loading environment variables and establishing a connection to the Vespa application.
        """
        load_dotenv()
        self.logger = logger

        if os.environ.get("USE_MTLS") == "true":
            self.logger.info("Connected using mTLS")
            mtls_key = os.environ.get("VESPA_CLOUD_MTLS_KEY")
            mtls_cert = os.environ.get("VESPA_CLOUD_MTLS_CERT")

            self.vespa_app_url = os.environ.get("VESPA_APP_MTLS_URL")
            if not self.vespa_app_url:
                raise ValueError(
                    "Please set the VESPA_APP_MTLS_URL environment variable"
                )

            if not mtls_cert or not mtls_key:
                raise ValueError(
                    "USE_MTLS was true, but VESPA_CLOUD_MTLS_KEY and VESPA_CLOUD_MTLS_CERT were not set"
                )

            # write the key and cert to a file
            mtls_key_path = "/tmp/vespa-data-plane-private-key.pem"
            with open(mtls_key_path, "w") as f:
                f.write(mtls_key)

            mtls_cert_path = "/tmp/vespa-data-plane-public-cert.pem"
            with open(mtls_cert_path, "w") as f:
                f.write(mtls_cert)

            # Instantiate Vespa connection
            self.app = Vespa(
                url=self.vespa_app_url, key=mtls_key_path, cert=mtls_cert_path
            )
        else:
            self.logger.info("Connected using token")
            self.vespa_app_url = os.environ.get("VESPA_APP_TOKEN_URL")
            if not self.vespa_app_url:
                raise ValueError(
                    "Please set the VESPA_APP_TOKEN_URL environment variable"
                )

            self.vespa_cloud_secret_token = os.environ.get("VESPA_CLOUD_SECRET_TOKEN")

            if not self.vespa_cloud_secret_token:
                raise ValueError(
                    "Please set the VESPA_CLOUD_SECRET_TOKEN environment variable"
                )

            # Instantiate Vespa connection
            self.app = Vespa(
                url=self.vespa_app_url,
                vespa_cloud_secret_token=self.vespa_cloud_secret_token,
            )

        self.app.wait_for_application_up()
        self.logger.info(f"Connected to Vespa at {self.vespa_app_url}")

    def get_fields(self, sim_map: bool = False):
        if not sim_map:
            return self.SELECT_FIELDS
        else:
            return "summaryfeatures"

    def format_query_results(
        self, query: str, response: VespaQueryResponse, hits: int = 5
    ) -> dict:
        """
        Format the Vespa query results.

        Args:
            query (str): The query text.
            response (VespaQueryResponse): The response from Vespa.
            hits (int, optional): Number of hits to display. Defaults to 5.

        Returns:
            dict: The JSON content of the response.
        """
        query_time = response.json.get("timing", {}).get("searchtime", -1)
        query_time = round(query_time, 2)
        count = response.json.get("root", {}).get("fields", {}).get("totalCount", 0)
        result_text = f"Query text: '{query}', query time {query_time}s, count={count}, top results:\n"
        self.logger.debug(result_text)
        return response.json

    async def query_vespa_bm25(
        self,
        query: str,
        q_emb: torch.Tensor,
        hits: int = 3,
        timeout: str = "10s",
        sim_map: bool = False,
        **kwargs,
    ) -> dict:
        """
        Query Vespa using the BM25 ranking profile.
        This corresponds to the "BM25" radio button in the UI.

        Args:
            query (str): The query text.
            q_emb (torch.Tensor): Query embeddings.
            hits (int, optional): Number of hits to retrieve. Defaults to 3.
            timeout (str, optional): Query timeout. Defaults to "10s".

        Returns:
            dict: The formatted query results.
        """
        async with self.app.asyncio(connections=1) as session:
            query_embedding = self.format_q_embs(q_emb)

            start = time.perf_counter()
            response: VespaQueryResponse = await session.query(
                body={
                    "yql": (
                        f"select {self.get_fields(sim_map=sim_map)} from {self.VESPA_SCHEMA_NAME} where userQuery();"
                    ),
                    "ranking": self.get_rank_profile("bm25", sim_map),
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
            self.logger.debug(
                f"Query time + data transfer took: {stop - start} s, Vespa reported searchtime was "
                f"{response.json.get('timing', {}).get('searchtime', -1)} s"
            )
        return self.format_query_results(query, response)

    def float_to_binary_embedding(self, float_query_embedding: dict) -> dict:
        """
        Convert float query embeddings to binary embeddings.

        Args:
            float_query_embedding (dict): Dictionary of float embeddings.

        Returns:
            dict: Dictionary of binary embeddings.
        """
        binary_query_embeddings = {}
        for key, vector in float_query_embedding.items():
            binary_vector = (
                np.packbits(np.where(np.array(vector) > 0, 1, 0))
                .astype(np.int8)
                .tolist()
            )
            binary_query_embeddings[key] = binary_vector
            if len(binary_query_embeddings) >= self.MAX_QUERY_TERMS:
                self.logger.warning(
                    f"Warning: Query has more than {self.MAX_QUERY_TERMS} terms. Truncating."
                )
                break
        return binary_query_embeddings

    def create_nn_query_strings(
        self, binary_query_embeddings: dict, target_hits_per_query_tensor: int = 20
    ) -> Tuple[str, dict]:
        """
        Create nearest neighbor query strings for Vespa.

        Args:
            binary_query_embeddings (dict): Binary query embeddings.
            target_hits_per_query_tensor (int, optional): Target hits per query tensor. Defaults to 20.

        Returns:
            Tuple[str, dict]: Nearest neighbor query string and query tensor dictionary.
        """
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

    def format_q_embs(self, q_embs: torch.Tensor) -> dict:
        """
        Convert query embeddings to a dictionary of lists.

        Args:
            q_embs (torch.Tensor): Query embeddings tensor.

        Returns:
            dict: Dictionary where each key is an index and value is the embedding list.
        """
        return {idx: emb.tolist() for idx, emb in enumerate(q_embs)}

    async def get_result_from_query(
        self,
        query: str,
        q_embs: torch.Tensor,
        ranking: str,
        idx_to_token: dict,
    ) -> Dict[str, Any]:
        """
        Get query results from Vespa based on the ranking method.

        Args:
            query (str): The query text.
            q_embs (torch.Tensor): Query embeddings.
            ranking (str): The ranking method to use.
            idx_to_token (dict): Index to token mapping.

        Returns:
            Dict[str, Any]: The query results.
        """

        # Remove stopwords from the query to avoid visual emphasis on irrelevant words (e.g., "the", "and", "of")
        query = backend.stopwords.filter(query)

        rank_method = ranking.split("_")[0]
        sim_map: bool = len(ranking.split("_")) > 1 and ranking.split("_")[1] == "sim"
        if rank_method == "colpali":  # ColPali
            result = await self.query_vespa_colpali(
                query=query, ranking=rank_method, q_emb=q_embs, sim_map=sim_map
            )
        elif rank_method == "hybrid":  # Hybrid ColPali+BM25
            result = await self.query_vespa_colpali(
                query=query, ranking=rank_method, q_emb=q_embs, sim_map=sim_map
            )
        elif rank_method == "bm25":
            result = await self.query_vespa_bm25(query, q_embs, sim_map=sim_map)
        else:
            raise ValueError(f"Unsupported ranking: {rank_method}")
        if "root" not in result or "children" not in result["root"]:
            result["root"] = {"children": []}
            return result
        for single_result in result["root"]["children"]:
            self.logger.debug(single_result["fields"].keys())
        return result

    def get_sim_maps_from_query(
        self, query: str, q_embs: torch.Tensor, ranking: str, idx_to_token: dict
    ):
        """
        Get similarity maps from Vespa based on the ranking method.

        Args:
            query (str): The query text.
            q_embs (torch.Tensor): Query embeddings.
            ranking (str): The ranking method to use.
            idx_to_token (dict): Index to token mapping.

        Returns:
            Dict[str, Any]: The query results.
        """
        # Get the result by calling asyncio.run
        result = asyncio.run(
            self.get_result_from_query(query, q_embs, ranking, idx_to_token)
        )
        vespa_sim_maps = []
        for single_result in result["root"]["children"]:
            vespa_sim_map = single_result["fields"].get("summaryfeatures", None)
            if vespa_sim_map is not None:
                vespa_sim_maps.append(vespa_sim_map)
            else:
                raise ValueError("No sim_map found in Vespa response")
        return vespa_sim_maps

    async def get_full_image_from_vespa(self, doc_id: str) -> str:
        """
        Retrieve the full image from Vespa for a given document ID.

        Args:
            doc_id (str): The document ID.

        Returns:
            str: The full image data.
        """
        async with self.app.asyncio(connections=1) as session:
            start = time.perf_counter()
            response: VespaQueryResponse = await session.query(
                body={
                    "yql": f'select full_image from {self.VESPA_SCHEMA_NAME} where id contains "{doc_id}"',
                    "ranking": "unranked",
                    "presentation.timing": True,
                    "ranking.matching.numThreadsPerSearch": 1,
                },
            )
            assert response.is_successful(), response.json
            stop = time.perf_counter()
            self.logger.debug(
                f"Getting image from Vespa took: {stop - start} s, Vespa reported searchtime was "
                f"{response.json.get('timing', {}).get('searchtime', -1)} s"
            )
        return response.json["root"]["children"][0]["fields"]["full_image"]

    def get_results_children(self, result: VespaQueryResponse) -> list:
        return result["root"]["children"]

    def results_to_search_results(
        self, result: VespaQueryResponse, idx_to_token: dict
    ) -> list:
        # Initialize sim_map_ fields in the result
        fields_to_add = [
            f"sim_map_{token}_{idx}"
            for idx, token in idx_to_token.items()
            if not SimMapGenerator.should_filter_token(token)
        ]
        for child in result["root"]["children"]:
            for sim_map_key in fields_to_add:
                child["fields"][sim_map_key] = None
        return self.get_results_children(result)

    async def get_suggestions(self, query: str) -> list:
        async with self.app.asyncio(connections=1) as session:
            start = time.perf_counter()
            yql = f'select questions from {self.VESPA_SCHEMA_NAME} where questions matches (".*{query}.*")'
            response: VespaQueryResponse = await session.query(
                body={
                    "yql": yql,
                    "query": query,
                    "ranking": "unranked",
                    "presentation.timing": True,
                    "presentation.summary": "suggestions",
                    "ranking.matching.numThreadsPerSearch": 1,
                },
            )
            assert response.is_successful(), response.json
            stop = time.perf_counter()
            self.logger.debug(
                f"Getting suggestions from Vespa took: {stop - start} s, Vespa reported searchtime was "
                f"{response.json.get('timing', {}).get('searchtime', -1)} s"
            )
            search_results = (
                response.json["root"]["children"]
                if "root" in response.json and "children" in response.json["root"]
                else []
            )
            questions = [
                result["fields"]["questions"]
                for result in search_results
                if "questions" in result["fields"]
            ]

            unique_questions = set([item for sublist in questions for item in sublist])

            # remove an artifact from our data generation
            if "string" in unique_questions:
                unique_questions.remove("string")

            return list(unique_questions)

    def get_rank_profile(self, ranking: str, sim_map: bool) -> str:
        if sim_map:
            return f"{ranking}_sim"
        else:
            return ranking

    async def query_vespa_colpali(
        self,
        query: str,
        ranking: str,
        q_emb: torch.Tensor,
        target_hits_per_query_tensor: int = 100,
        hnsw_explore_additional_hits: int = 300,
        hits: int = 3,
        timeout: str = "10s",
        sim_map: bool = False,
        **kwargs,
    ) -> dict:
        """
        Query Vespa using nearest neighbor search with mixed tensors for MaxSim calculations.
        This corresponds to the "ColPali" radio button in the UI.

        Args:
            query (str): The query text.
            q_emb (torch.Tensor): Query embeddings.
            target_hits_per_query_tensor (int, optional): Target hits per query tensor. Defaults to 20.
            hits (int, optional): Number of hits to retrieve. Defaults to 3.
            timeout (str, optional): Query timeout. Defaults to "10s".

        Returns:
            dict: The formatted query results.
        """
        async with self.app.asyncio(connections=1) as session:
            float_query_embedding = self.format_q_embs(q_emb)
            binary_query_embeddings = self.float_to_binary_embedding(
                float_query_embedding
            )

            # Mixed tensors for MaxSim calculations
            query_tensors = {
                "input.query(qtb)": binary_query_embeddings,
                "input.query(qt)": float_query_embedding,
            }
            nn_string, nn_query_dict = self.create_nn_query_strings(
                binary_query_embeddings, target_hits_per_query_tensor
            )
            query_tensors.update(nn_query_dict)
            response: VespaQueryResponse = await session.query(
                body={
                    **query_tensors,
                    "presentation.timing": True,
                    "yql": (
                        f"select {self.get_fields(sim_map=sim_map)} from {self.VESPA_SCHEMA_NAME} where {nn_string} or userQuery()"
                    ),
                    "ranking.profile": self.get_rank_profile(
                        ranking=ranking, sim_map=sim_map
                    ),
                    "timeout": timeout,
                    "hits": hits,
                    "query": query,
                    "hnsw.exploreAdditionalHits": hnsw_explore_additional_hits,
                    "ranking.rerankCount": 100,
                    **kwargs,
                },
            )
            assert response.is_successful(), response.json
        return self.format_query_results(query, response)

    async def keepalive(self) -> bool:
        """
        Query Vespa to keep the connection alive.

        Returns:
            bool: True if the connection is alive.
        """
        async with self.app.asyncio(connections=1) as session:
            response: VespaQueryResponse = await session.query(
                body={
                    "yql": f"select title from {self.VESPA_SCHEMA_NAME} where true limit 1;",
                    "ranking": "unranked",
                    "query": "keepalive",
                    "timeout": "3s",
                    "hits": 1,
                },
            )
            assert response.is_successful(), response.json
        return True
