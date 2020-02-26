import json
import os
import streamlit as st
import plotly as py
import plotly.figure_factory as ff
from random import sample
from msmarco import load_msmarco_queries, load_msmarco_qrels, extract_querie_relevance
from embedding import create_document_embedding
from pandas import DataFrame
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from experiments import evaluate, create_vespa_body_request, vespa_search

os.environ["TFHUB_CACHE_DIR"] = "data/models"

QUERIES_FILE_PATH = "data/msmarco/train_test_set/msmarco-doctest-queries.tsv.gz"
RELEVANCE_FILE_PATH = "data/msmarco/train_test_set/msmarco-doctest-qrels.tsv.gz"
RANK_PROFILE_OPTIONS = (
    "BM25",
    "Native Rank",
    "embedding(title) + embedding(body)",
    "BM25 + embedding(title) + embedding(body)",
)
# todo: think how I am going to encode and present Scaled ranking functions
# "Scaled (AND) BM25 + title and body gse": "listwise_linear_bm25_gse_title_body_and",
# "Scaled (OR) BM25 + title and body gse": "listwise_linear_bm25_gse_title_body_or",


RANK_PROFILE_MAP = {
    "BM25": "bm25",
    "Native Rank": "default",
    "embedding(title) + embedding(body)": {
        "word2vec": "word2vec_title_body_all",
        "gse": "gse_title_body_all",
        "bert": "bert_title_body_all",
    },
    "BM25 + embedding(title) + embedding(body)": {
        "word2vec": "bm25_word2vec_title_body_all",
        "gse": "bm25_gse_title_body_all",
        "bert": "bm25_bert_title_body_all",
    },
}
# todo: I think I dont need RANK_PROFILE_EMBEDDING
# RANK_PROFILE_EMBEDDING = {
#     "bm25": None,
#     "default": None,
#     "word2vec_title_body_all": "word2vec",
#     "bm25_word2vec_title_body_all": "word2vec",
#     "gse_title_body_all": "gse",
#     "bm25_gse_title_body_all": "gse",
#     "listwise_linear_bm25_gse_title_body_and": "gse",
#     "listwise_linear_bm25_gse_title_body_or": "gse",
#     "bert_title_body_all": "bert",
#     "bm25_bert_title_body_all": "bert",
# }
GRAMMAR_OPTIONS = ["None", "AND", "OR", "weakAND"]
GRAMMAR_OPERATOR_MAP = {"AND": False, "OR": True}
EMBEDDING_OPTIONS = ["word2vec", "gse", "bert"]
ANN_OPTIONS = ["None", "title", "body", "title_body"]
LIMIT_HITS_GRAPH = 10


def get_rank_profile(rank_profile, embedding):
    if "embedding" in rank_profile:
        return RANK_PROFILE_MAP[rank_profile][embedding]
    else:
        return RANK_PROFILE_MAP[rank_profile]


@st.cache(ignore_hash=True)
def retrieve_model(model_type):
    if model_type == "word2vec":
        return {
            "model": hub.load(
                "https://tfhub.dev/google/Wiki-words-500-with-normalization/2"
            ),
            "model_source": "tf_hub",
        }
    elif model_type == "gse":
        return {
            "model": hub.load("https://tfhub.dev/google/universal-sentence-encoder/4"),
            "model_source": "tf_hub",
        }
    elif model_type == "bert":
        return {
            "model": SentenceTransformer("distilbert-base-nli-stsb-mean-tokens"),
            "model_source": "bert",
        }


def create_experiment_file_name(rank_profile, grammar_operator, ann, embedding, hits):
    file_name = "grammar_{}_ann_{}_rank_{}_embedding_{}_hits_{}".format(
        grammar_operator,
        ann,
        get_rank_profile(rank_profile, embedding),
        embedding,
        hits,
    )
    return file_name


def compute_all_options(
    vespa_url,
    vespa_port,
    output_dir,
    rank_profiles,
    grammar_operators,
    ann_operators,
    embeddings,
    hits,
):
    query_relevance = sample_query_relevance_data(number_queries=None)
    for rank_profile in rank_profiles:
        for grammar_operator in grammar_operators:
            grammar_operator = None if grammar_operator is "None" else grammar_operator
            for ann in ann_operators:
                ann = None if ann is "None" else ann
                for embedding in embeddings:
                    file_name = create_experiment_file_name(
                        rank_profile, grammar_operator, ann, embedding, hits
                    )
                    file_path = os.path.join(output_dir, file_name)
                    if not os.path.exists(file_path):
                        model1 = retrieve_model(embedding)
                        try:
                            records, aggregate_metrics, position_freq = evaluate(
                                query_relevance=query_relevance,
                                parsed_rank_profile=get_rank_profile(
                                    rank_profile, embedding
                                ),
                                grammar_operator=grammar_operator,
                                ann_operator=ann,
                                embedding_type=embedding,
                                vespa_url=vespa_url,
                                vespa_port=vespa_port,
                                hits=int(hits),
                                model=model1,
                            )
                        except ValueError as e:
                            print(str(e))
                            continue
                        with open(file_path, "w") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "aggregate_metrics": aggregate_metrics,
                                        "position_freq": position_freq,
                                    }
                                )
                            )


def load_all_options(
    output_dir, rank_profiles, grammar_operators, ann_operators, embeddings, hits
):
    results = []
    for rank_profile in rank_profiles:
        for grammar_operator in grammar_operators:
            for ann in ann_operators:
                for embedding in embeddings:
                    file_name = create_experiment_file_name(
                        rank_profile, grammar_operator, ann, embedding, hits
                    )
                    file_path = os.path.join(output_dir, file_name)
                    try:
                        result = json.load(open(file_path, "r"))
                    except FileNotFoundError:
                        continue
                    result.update(
                        {
                            "rank_profile": rank_profile,
                            "grammar_operator": grammar_operator,
                            "ann_operator": ann,
                            "embedding_type": embedding,
                        }
                    )
                    results.append(result)
    return results


def main():
    vespa_url = st.sidebar.text_input("Vespa url", "http://localhost")
    vespa_port = st.sidebar.text_input("Vespa port", 8080)

    page = st.sidebar.selectbox(
        "Choose a page",
        ["Simple query", "Ranking function comparison", "Results summary", "Report"],
    )

    if page == "Simple query":
        page_simple_query_page(vespa_url=vespa_url, vespa_port=vespa_port)
    # elif page == "Ranking function comparison":
    #     page_ranking_function_comparison(vespa_url=vespa_url, vespa_port=vespa_port)
    elif page == "Results summary":
        page_results_summary(vespa_url=vespa_url, vespa_port=vespa_port)


def page_results_summary(vespa_url, vespa_port):

    grammar_operators = st.multiselect("Choose grammar operators", GRAMMAR_OPTIONS)
    ann_operators = st.multiselect("ANN operator", ANN_OPTIONS)
    rank_profiles = st.multiselect("Choose rank profiles", RANK_PROFILE_OPTIONS)
    embeddings = st.multiselect("Embedding type", EMBEDDING_OPTIONS)
    output_dir = "data/msmarco/experiments"

    if st.button("Evaluate"):

        hits = 100

        compute_all_options(
            vespa_url,
            vespa_port,
            output_dir,
            rank_profiles,
            grammar_operators,
            ann_operators,
            embeddings,
            hits,
        )

        results = load_all_options(
            output_dir,
            rank_profiles,
            grammar_operators,
            ann_operators,
            embeddings,
            hits,
        )

        position_freqs = []
        ranking_names = []
        results_summary = []
        for result in results:
            position_freqs.append(result["position_freq"])
            ranking_names.append(result["aggregate_metrics"]["rank_name"])
            results_summary.append(
                {
                    "rank_name": result["aggregate_metrics"]["rank_name"],
                    "rank_profile": result["rank_profile"],
                    "grammar_operator": result["grammar_operator"],
                    "ann_operator": result["ann_operator"],
                    "embedding_type": result["embedding_type"],
                    "number_queries": result["aggregate_metrics"]["number_queries"],
                    "qps": result["aggregate_metrics"]["qps"],
                    "mrr": result["aggregate_metrics"]["mrr"],
                    "recall": result["aggregate_metrics"]["recall"],
                    "average_matched": result["aggregate_metrics"]["average_matched"],
                }
            )

        display_results(position_freqs, ranking_names, results_summary, hits)


def display_results(
    position_freqs, ranking_names, results_summary, hits, display_graph=True
):
    if display_graph:
        hits = min(hits, LIMIT_HITS_GRAPH)
        z = [list(x) for x in zip(*position_freqs)]
        z_text = z
        x = ranking_names
        y = [str(x + 1) for x in range(int(hits))]

        fig = ff.create_annotated_heatmap(
            z, x=x, y=y, annotation_text=z_text, colorscale=py.colors.diverging.RdYlGn
        )
        fig.update_layout(
            xaxis_title_text="Rank profile",  # xaxis label
            yaxis_title_text="Position",  # yaxis label
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig)

    st.write(
        DataFrame.from_records(results_summary).sort_values(by="mrr", ascending=False)
    )


# def page_ranking_function_comparison(vespa_url, vespa_port):
#     rank_profile_1 = st.sidebar.selectbox(
#         "Ranking 1: rank profile", RANK_PROFILE_OPTIONS
#     )
#     grammar_operator_1 = st.sidebar.selectbox("Ranking 1: Grammar", ("AND", "OR"))
#     ann_operator_1 = None
#     if RANK_PROFILE_EMBEDDING[RANK_PROFILE_MAP[rank_profile_1]] in EMBEDDING_OPTIONS:
#         ann_operator_1 = st.sidebar.selectbox(
#             "Ranking 1: ANN operator", (None, "title", "body", "title_body")
#         )
#     rank_profile_2 = st.sidebar.selectbox(
#         "Ranking 2: rank profile", RANK_PROFILE_OPTIONS
#     )
#     grammar_operator_2 = st.sidebar.selectbox("Ranking 2: Grammar", ("AND", "OR"))
#     ann_operator_2 = None
#     if RANK_PROFILE_EMBEDDING[RANK_PROFILE_MAP[rank_profile_2]] in EMBEDDING_OPTIONS:
#         ann_operator_2 = st.sidebar.selectbox(
#             "Ranking 2: ANN operator", (None, "title", "body", "title_body")
#         )
#     number_queries = int(st.text_input("Number of queries to send", "20"))
#
#     hits = int(st.text_input("Number of hits to evaluate per query", "10"))
#
#     if st.button("Evaluate"):
#         query_relevance = sample_query_relevance_data(number_queries=number_queries)
#
#         model1 = retrieve_model(
#             RANK_PROFILE_EMBEDDING[RANK_PROFILE_MAP[rank_profile_1]]
#         )
#         records_1, aggregate_metrics_1, position_freq_1 = evaluate(
#             query_relevance=query_relevance,
#             parsed_rank_profile=RANK_PROFILE_MAP[rank_profile_1],
#             grammar_operator=GRAMMAR_OPERATOR_MAP[grammar_operator_1],
#             vespa_url=vespa_url,
#             vespa_port=vespa_port,
#             hits=int(hits),
#             model=model1,
#             ann=ann_operator_1,
#         )
#
#         model2 = retrieve_model(
#             RANK_PROFILE_EMBEDDING[RANK_PROFILE_MAP[rank_profile_2]]
#         )
#         records_2, aggregate_metrics_2, position_freq_2 = evaluate(
#             query_relevance=query_relevance,
#             parsed_rank_profile=RANK_PROFILE_MAP[rank_profile_2],
#             grammar_operator=GRAMMAR_OPERATOR_MAP[grammar_operator_2],
#             vespa_url=vespa_url,
#             vespa_port=vespa_port,
#             hits=int(hits),
#             model=model2,
#             ann=ann_operator_2,
#         )
#         position_freqs = [position_freq_1, position_freq_2]
#         ranking_names = [
#             aggregate_metrics_1["rank_name"],
#             aggregate_metrics_2["rank_name"],
#         ]
#         results_summary = [aggregate_metrics_1, aggregate_metrics_2]
#
#         display_results(position_freqs, ranking_names, results_summary, hits)


def page_simple_query_page(vespa_url, vespa_port):
    predefined_queries = st.checkbox("Use pre-defined queries")

    if predefined_queries:
        query_relevance = sample_query_relevance_data(number_queries=5)
        query_relevance = {
            query: relevant_id for _, (query, relevant_id) in query_relevance.items()
        }
        query = st.selectbox("Choose a query", list(query_relevance.keys()))
    else:
        query = st.text_input("Query", "")

    st.markdown("---")

    grammar_operator = st.selectbox("Choose grammar operators", GRAMMAR_OPTIONS)
    grammar_operator = None if grammar_operator is "None" else grammar_operator
    if grammar_operator is None:
        available_ann_options = [x for x in ANN_OPTIONS if x is not "None"]
    else:
        available_ann_options = ANN_OPTIONS
    ann_operator = st.selectbox("ANN operator", available_ann_options)
    ann_operator = None if ann_operator is "None" else ann_operator
    rank_profile = st.selectbox("Choose rank profiles", RANK_PROFILE_OPTIONS)
    if "embedding" in rank_profile or ann_operator is not None:
        embedding = st.selectbox("Embedding type", EMBEDDING_OPTIONS)
    else:
        embedding = None

    embedding_vector = None
    if embedding in EMBEDDING_OPTIONS:
        model = retrieve_model(embedding)
        embedding_vector = create_document_embedding(
            text=query,
            model=model["model"],
            model_source=model["model_source"],
            normalize=True,
        )

    st.markdown("---")

    if query != "":
        print_request_body = st.checkbox("Print request body?")

        debug = st.checkbox("Debug?")

        output_format = st.radio(
            "Select output format", ("parsed vespa results", "raw vespa results")
        )

        tracelevel = None
        trace = st.checkbox("Specify tracelevel?")
        if trace:
            tracelevel = st.selectbox("Tracelevel", [3, 9], 0)

        request_body = create_vespa_body_request(
            query=query,
            parsed_rank_profile=get_rank_profile(
                rank_profile=rank_profile, embedding=embedding
            ),
            grammar_operator=grammar_operator,
            ann_operator=ann_operator,
            embedding_type=embedding,
            hits=10,
            embedding_vector=embedding_vector,
            tracelevel=tracelevel,
        )
        search_results = vespa_search(
            vespa_url=vespa_url, vespa_port=vespa_port, body=request_body
        )

        #
        # Debug
        #
        if debug:
            if "children" in search_results["root"]:
                debug_data = []
                for hit in search_results["root"]["children"]:
                    debug_data.append(
                        {
                            "complete_id": hit["id"],
                            "id": hit["fields"]["id"],
                            "title_dot_product": hit["fields"]["rankfeatures"].get(
                                "rankingExpression(dot_product_title)"
                            ),
                            "body_dot_product": hit["fields"]["rankfeatures"].get(
                                "rankingExpression(dot_product_body)"
                            ),
                        }
                    )
                st.write(DataFrame.from_records(debug_data))

        st.markdown("---")
        if print_request_body:
            st.write(request_body)
        if output_format == "raw vespa results":
            st.markdown("## Showing raw results")
            st.write(search_results)
        elif output_format == "parsed vespa results":
            st.markdown("## Showing parsed results")
            st.markdown("### Click to see more")
            results_title = {}
            if "children" in search_results["root"]:
                for hit in search_results["root"]["children"]:
                    if (
                        predefined_queries
                        and hit["fields"]["id"] == query_relevance[query]
                    ):
                        results_title["*** " + hit["fields"]["title"] + " ***"] = {
                            "url": hit["fields"]["url"],
                            "body": hit["fields"]["body"],
                            "relevance": hit["relevance"],
                            "id": hit["id"],
                        }

                    else:
                        results_title[hit["fields"]["title"]] = {
                            "url": hit["fields"]["url"],
                            "body": hit["fields"]["body"],
                            "relevance": hit["relevance"],
                            "id": hit["id"],
                        }
                for title in results_title:
                    if st.checkbox(title):
                        st.markdown(
                            "* relevance: {}".format(results_title[title]["relevance"])
                        )
                        st.markdown("* docid: {}".format(results_title[title]["id"]))
                        st.markdown("* url: {}".format(results_title[title]["url"]))
                        st.markdown("* text:")
                        st.write(results_title[title]["body"])
            else:
                st.markdown("## No hits available")


@st.cache()
def sample_query_relevance_data(number_queries):
    queries = load_msmarco_queries(queries_file_path=QUERIES_FILE_PATH)
    qrels = load_msmarco_qrels(relevance_file_path=RELEVANCE_FILE_PATH)
    if number_queries is not None:
        qrels = {k: qrels[k] for k in sample(list(qrels), number_queries)}
    query_relevance = extract_querie_relevance(qrels, queries)
    return query_relevance


if __name__ == "__main__":
    main()
