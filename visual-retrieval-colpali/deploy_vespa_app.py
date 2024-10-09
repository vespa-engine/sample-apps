#!/usr/bin/env python3

import argparse
from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    HNSW,
    RankProfile,
    Function,
    AuthClient,
    Parameter,
    FieldSet,
    SecondPhaseRanking,
)
from vespa.deployment import VespaCloud
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Deploy Vespa application")
    parser.add_argument("--tenant_name", required=True, help="Vespa Cloud tenant name")
    parser.add_argument(
        "--vespa_application_name", required=True, help="Vespa application name"
    )
    parser.add_argument(
        "--token_id_write", required=True, help="Vespa Cloud token ID for write access"
    )
    parser.add_argument(
        "--token_id_read", required=True, help="Vespa Cloud token ID for read access"
    )

    args = parser.parse_args()
    tenant_name = args.tenant_name
    vespa_app_name = args.vespa_application_name
    token_id_write = args.token_id_write
    token_id_read = args.token_id_read

    # Define the Vespa schema
    colpali_schema = Schema(
        name="pdf_page",
        document=Document(
            fields=[
                Field(
                    name="id",
                    type="string",
                    indexing=["summary", "index"],
                    match=["word"],
                ),
                Field(name="url", type="string", indexing=["summary", "index"]),
                Field(
                    name="title",
                    type="string",
                    indexing=["summary", "index"],
                    match=["text"],
                    index="enable-bm25",
                ),
                Field(
                    name="page_number", type="int", indexing=["summary", "attribute"]
                ),
                Field(name="image", type="raw", indexing=["summary"]),
                Field(name="full_image", type="raw", indexing=["summary"]),
                Field(
                    name="text",
                    type="string",
                    indexing=["summary", "index"],
                    match=["text"],
                    index="enable-bm25",
                ),
                Field(
                    name="embedding",
                    type="tensor<int8>(patch{}, v[16])",
                    indexing=[
                        "attribute",
                        "index",
                    ],  # adds HNSW index for candidate retrieval.
                    ann=HNSW(
                        distance_metric="hamming",
                        max_links_per_node=32,
                        neighbors_to_explore_at_insert=400,
                    ),
                ),
            ]
        ),
        fieldsets=[
            FieldSet(name="default", fields=["title", "url", "page_number", "text"]),
            FieldSet(name="image", fields=["image"]),
        ],
    )

    # Define rank profiles
    colpali_profile = RankProfile(
        name="default",
        inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
        functions=[
            Function(
                name="max_sim",
                expression="""
                    sum(
                        reduce(
                            sum(
                                query(qt) * unpack_bits(attribute(embedding)) , v
                            ),
                            max, patch
                        ),
                        querytoken
                    )
                """,
            ),
            Function(name="bm25_score", expression="bm25(title) + bm25(text)"),
        ],
        first_phase="bm25_score",
        second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=10),
    )
    colpali_schema.add_rank_profile(colpali_profile)

    # Add retrieval-and-rerank rank profile
    input_query_tensors = []
    MAX_QUERY_TERMS = 64
    for i in range(MAX_QUERY_TERMS):
        input_query_tensors.append((f"query(rq{i})", "tensor<int8>(v[16])"))

    input_query_tensors.append(("query(qt)", "tensor<float>(querytoken{}, v[128])"))
    input_query_tensors.append(("query(qtb)", "tensor<int8>(querytoken{}, v[16])"))

    colpali_retrieval_profile = RankProfile(
        name="retrieval-and-rerank",
        inputs=input_query_tensors,
        functions=[
            Function(
                name="max_sim",
                expression="""
                    sum(
                        reduce(
                            sum(
                                query(qt) * unpack_bits(attribute(embedding)) , v
                            ),
                            max, patch
                        ),
                        querytoken
                    )
                """,
            ),
            Function(
                name="max_sim_binary",
                expression="""
                    sum(
                      reduce(
                        1/(1 + sum(
                            hamming(query(qtb), attribute(embedding)) ,v)
                        ),
                        max,
                        patch
                      ),
                      querytoken
                    )
                """,
            ),
        ],
        first_phase="max_sim_binary",
        second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=10),
    )
    colpali_schema.add_rank_profile(colpali_retrieval_profile)

    # Create the Vespa application package
    vespa_application_package = ApplicationPackage(
        name=vespa_app_name,
        schema=[colpali_schema],
        auth_clients=[
            AuthClient(
                id="mtls",  # Note that you still need to include the mtls client.
                permissions=["read", "write"],
                parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
            ),
            AuthClient(
                id="token_write",
                permissions=["read", "write"],
                parameters=[Parameter("token", {"id": token_id_write})],
            ),
            AuthClient(
                id="token_read",
                permissions=["read"],
                parameters=[Parameter("token", {"id": token_id_read})],
            ),
        ],
    )
    vespa_team_api_key = os.getenv("VESPA_TEAM_API_KEY")
    # Deploy the application to Vespa Cloud
    vespa_cloud = VespaCloud(
        tenant=tenant_name,
        application=vespa_app_name,
        key_content=vespa_team_api_key,
        application_root="colpali-with-snippets",
        #application_package=vespa_application_package,
    )

    #app = vespa_cloud.deploy()
    vespa_cloud.deploy_from_disk("default", "colpali-with-snippets")

    # Output the endpoint URL
    endpoint_url = vespa_cloud.get_token_endpoint()
    print(f"Application deployed. Token endpoint URL: {endpoint_url}")


if __name__ == "__main__":
    main()
