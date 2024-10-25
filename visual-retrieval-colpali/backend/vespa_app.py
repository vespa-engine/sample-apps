import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from vespa.application import Vespa
from vespa.io import VespaQueryResponse


class VespaQueryClient:
    MAX_QUERY_TERMS = 64
    VESPA_SCHEMA_NAME = "pdf_page"
    SELECT_FIELDS = "id,title,url,blur_image,page_number,snippet,text,summaryfeatures"

    def __init__(self):
        """
        Initialize the VespaQueryClient by loading environment variables and establishing a connection to the Vespa application.
        """
        load_dotenv()

        if os.environ.get("USE_MTLS") == "true":
            print("Connected using mTLS")
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
            print("Connected using token")
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
        print(f"Connected to Vespa at {self.vespa_app_url}")

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
        print(result_text)
        return response.json

    async def query_vespa_default(
        self,
        query: str,
        q_emb: torch.Tensor,
        hits: int = 3,
        timeout: str = "10s",
        **kwargs,
    ) -> dict:
        """
        Query Vespa using the default ranking profile.

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
                        f"select {self.SELECT_FIELDS} from {self.VESPA_SCHEMA_NAME} where userQuery();"
                    ),
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
                f"Query time + data transfer took: {stop - start} s, Vespa reported searchtime was "
                f"{response.json.get('timing', {}).get('searchtime', -1)} s"
            )
        return self.format_query_results(query, response)

    async def query_vespa_bm25(
        self,
        query: str,
        q_emb: torch.Tensor,
        hits: int = 3,
        timeout: str = "10s",
        **kwargs,
    ) -> dict:
        """
        Query Vespa using the BM25 ranking profile.

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
                        f"select {self.SELECT_FIELDS} from {self.VESPA_SCHEMA_NAME} where userQuery();"
                    ),
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
                print(
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
        token_to_idx: dict,
    ) -> Dict[str, Any]:
        """
        Get query results from Vespa based on the ranking method.

        Args:
            query (str): The query text.
            q_embs (torch.Tensor): Query embeddings.
            ranking (str): The ranking method to use.
            token_to_idx (dict): Token to index mapping.

        Returns:
            Dict[str, Any]: The query results.
        """
        print(query)
        print(token_to_idx)

        if ranking == "nn+colpali":
            result = await self.query_vespa_nearest_neighbor(query, q_embs)
        elif ranking == "bm25+colpali":
            result = await self.query_vespa_default(query, q_embs)
        elif ranking == "bm25":
            result = await self.query_vespa_bm25(query, q_embs)
        else:
            raise ValueError(f"Unsupported ranking: {ranking}")

        # Print score, title id, and text of the results
        if "root" not in result or "children" not in result["root"]:
            result["root"] = {"children": []}
            return result
        for idx, child in enumerate(result["root"]["children"]):
            print(
                f"Result {idx+1}: {child['relevance']}, {child['fields']['title']}, {child['fields']['id']}"
            )
        for single_result in result["root"]["children"]:
            print(single_result["fields"].keys())
        return result

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
                },
            )
            assert response.is_successful(), response.json
            stop = time.perf_counter()
            print(
                f"Getting image from Vespa took: {stop - start} s, Vespa reported searchtime was "
                f"{response.json.get('timing', {}).get('searchtime', -1)} s"
            )
        return response.json["root"]["children"][0]["fields"]["full_image"]

    async def get_suggestions(self, query: str) -> list:
        async with self.app.asyncio(connections=1) as session:
            start = time.perf_counter()
            yql = f'select questions from {self.VESPA_SCHEMA_NAME} where questions matches "{query}" limit 3'
            response: VespaQueryResponse = await session.query(
                body={
                    "yql": yql,
                    "ranking": "unranked",
                    "presentation.timing": True,
                    "presentation.summary": "suggestions",
                },
            )
            assert response.is_successful(), response.json
            stop = time.perf_counter()
            print(
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
            flat_questions = [item for sublist in questions for item in sublist]
            return flat_questions

    async def query_vespa_nearest_neighbor(
        self,
        query: str,
        q_emb: torch.Tensor,
        target_hits_per_query_tensor: int = 20,
        hits: int = 3,
        timeout: str = "10s",
        **kwargs,
    ) -> dict:
        """
        Query Vespa using nearest neighbor search with mixed tensors for MaxSim calculations.

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
                        f"select {self.SELECT_FIELDS} from {self.VESPA_SCHEMA_NAME} where {nn_string} or userQuery()"
                    ),
                    "ranking.profile": "retrieval-and-rerank",
                    "timeout": timeout,
                    "hits": hits,
                    "query": query,
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
