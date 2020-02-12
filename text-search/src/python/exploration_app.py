import json
import os
import streamlit as st
import plotly as py
import plotly.figure_factory as ff
from requests import post
from time import time
from random import sample
from msmarco import load_msmarco_queries, load_msmarco_qrels, extract_querie_relevance
from embedding import create_document_embedding
from pandas import DataFrame
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer

QUERIES_FILE_PATH = "data/msmarco/train_test_set/msmarco-doctest-queries.tsv.gz"
RELEVANCE_FILE_PATH = "data/msmarco/train_test_set/msmarco-doctest-qrels.tsv.gz"
RANK_PROFILE_OPTIONS = (
    "BM25",
    "Native Rank",
    "Title and body word2vec",
    "BM25 + title and body word2vec",
    "Title and body gse",
    "BM25 + title and body gse",
    "Title and body bert",
    "BM25 + title and body bert",
    "Scaled (AND) BM25 + title and body gse",
    "Scaled (OR) BM25 + title and body gse",
)
RANK_PROFILE_MAP = {
    "BM25": "bm25",
    "Native Rank": "default",
    "Title and body word2vec": "word2vec_title_body_all",
    "BM25 + title and body word2vec": "bm25_word2vec_title_body_all",
    "Title and body gse": "gse_title_body_all",
    "BM25 + title and body gse": "bm25_gse_title_body_all",
    "Scaled (AND) BM25 + title and body gse": "listwise_linear_bm25_gse_title_body_and",
    "Scaled (OR) BM25 + title and body gse": "listwise_linear_bm25_gse_title_body_or",
    "Title and body bert": "bert_title_body_all",
    "BM25 + title and body bert": "bm25_bert_title_body_all",
}
RANK_PROFILE_EMBEDDING = {
    "bm25": None,
    "default": None,
    "word2vec_title_body_all": "word2vec",
    "bm25_word2vec_title_body_all": "word2vec",
    "gse_title_body_all": "gse",
    "bm25_gse_title_body_all": "gse",
    "listwise_linear_bm25_gse_title_body_and": "gse",
    "listwise_linear_bm25_gse_title_body_or": "gse",
    "bert_title_body_all": "bert",
    "bm25_bert_title_body_all": "bert",
}
AVAILABLE_EMBEDDINGS = ["word2vec", "gse", "bert"]
GRAMMAR_OPERATOR_MAP = {"AND": False, "OR": True}
LIMIT_HITS_GRAPH = 10


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


def main():
    vespa_url = st.sidebar.text_input("Vespa url", "http://localhost")
    vespa_port = st.sidebar.text_input("Vespa port", 8080)

    page = st.sidebar.selectbox(
        "Choose a page",
        ["Simple query", "Ranking function comparison", "Results summary", "Report"],
    )

    if page == "Simple query":
        page_simple_query_page(vespa_url=vespa_url, vespa_port=vespa_port)
    elif page == "Ranking function comparison":
        page_ranking_function_comparison(vespa_url=vespa_url, vespa_port=vespa_port)
    elif page == "Results summary":
        page_results_summary(vespa_url=vespa_url, vespa_port=vespa_port)
    elif page == "Report":
        page_report(vespa_url=vespa_url, vespa_port=vespa_port)


def compute_all_options(
    vespa_url,
    vespa_port,
    output_dir,
    rank_profiles,
    grammar_operators,
    ann_operators,
    hits,
):
    query_relevance = sample_query_relevance_data(number_queries=None)
    for rank_profile in rank_profiles:
        for grammar_operator in grammar_operators:
            for ann in ann_operators:
                # if (
                #     RANK_PROFILE_EMBEDDING[RANK_PROFILE_MAP[rank_profile]] is None
                #     and ann
                # ):
                #     continue
                file_name = "full_evaluation_{}_{}_ANN_{}_hits_{}".format(
                    RANK_PROFILE_MAP[rank_profile], grammar_operator, ann, hits
                )
                file_path = os.path.join(output_dir, file_name)
                if not os.path.exists(file_path):
                    model1 = retrieve_model(
                        RANK_PROFILE_EMBEDDING[RANK_PROFILE_MAP[rank_profile]]
                    )
                    records, aggregate_metrics, position_freq = evaluate(
                        query_relevance=query_relevance,
                        rank_profile=RANK_PROFILE_MAP[rank_profile],
                        grammar_any=GRAMMAR_OPERATOR_MAP[grammar_operator],
                        vespa_url=vespa_url,
                        vespa_port=vespa_port,
                        hits=int(hits),
                        model=model1,
                        ann=ann,
                    )
                    with open(file_path, "w") as f:
                        f.write(
                            json.dumps(
                                {
                                    "aggregate_metrics": aggregate_metrics,
                                    "position_freq": position_freq,
                                }
                            )
                        )


def load_all_options(output_dir, rank_profiles, grammar_operators, ann_operators, hits):
    results = []
    for rank_profile in rank_profiles:
        for grammar_operator in grammar_operators:
            for ann in ann_operators:
                # if (
                #     RANK_PROFILE_EMBEDDING[RANK_PROFILE_MAP[rank_profile]] is None
                #     and ann
                # ):
                #     continue
                file_name = "full_evaluation_{}_{}_ANN_{}_hits_{}".format(
                    RANK_PROFILE_MAP[rank_profile], grammar_operator, ann, hits
                )
                file_path = os.path.join(output_dir, file_name)
                results.append(json.load(open(file_path, "r")))
    return results


def page_results_summary(vespa_url, vespa_port):

    rank_profiles = st.multiselect("Choose rank profiles", RANK_PROFILE_OPTIONS)
    grammar_operators = st.multiselect("Choose grammar operators", ["AND", "OR"])
    ann_operators = st.multiselect(
        "ANN operator", [None, "title", "body", "title_body"]
    )
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
            hits,
        )

        results = load_all_options(
            output_dir, rank_profiles, grammar_operators, ann_operators, hits
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
                    "number_queries": result["aggregate_metrics"]["number_queries"],
                    "qps": result["aggregate_metrics"]["qps"],
                    "mrr": result["aggregate_metrics"]["mrr"],
                    "recall": result["aggregate_metrics"]["recall"],
                    "average_matched": result["aggregate_metrics"]["average_matched"],
                }
            )

        display_results(position_freqs, ranking_names, results_summary, hits)


def page_report(vespa_url, vespa_port):

    # rank_profiles = st.multiselect("Choose rank profiles", RANK_PROFILE_OPTIONS)
    # grammar_operators = st.multiselect("Choose grammar operators", ["AND", "OR"])
    # ann_operators = st.multiselect(
    #     "ANN operator", [None, "title", "body", "title_body"]
    # )

    st.markdown("## AND operator")

    and_rank_profiles = st.multiselect("Choose rank profiles", RANK_PROFILE_OPTIONS)
    and_grammar_operators = ["AND"]
    and_ann_operators = [None]

    output_dir = "data/msmarco/experiments"

    if st.button("Evaluate"):

        hits = 100

        compute_all_options(
            vespa_url,
            vespa_port,
            output_dir,
            and_rank_profiles,
            and_grammar_operators,
            and_ann_operators,
            hits,
        )

        results = load_all_options(
            output_dir,
            and_rank_profiles,
            and_grammar_operators,
            and_ann_operators,
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
                    "number_queries": result["aggregate_metrics"]["number_queries"],
                    "qps": result["aggregate_metrics"]["qps"],
                    "mrr": result["aggregate_metrics"]["mrr"],
                    "recall": result["aggregate_metrics"]["recall"],
                }
            )

        display_results(
            position_freqs, ranking_names, results_summary, hits, display_graph=False
        )


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


def page_ranking_function_comparison(vespa_url, vespa_port):
    rank_profile_1 = st.sidebar.selectbox(
        "Ranking 1: rank profile", RANK_PROFILE_OPTIONS
    )
    grammar_operator_1 = st.sidebar.selectbox("Ranking 1: Grammar", ("AND", "OR"))
    ann_operator_1 = None
    if RANK_PROFILE_EMBEDDING[RANK_PROFILE_MAP[rank_profile_1]] in AVAILABLE_EMBEDDINGS:
        ann_operator_1 = st.sidebar.selectbox(
            "Ranking 1: ANN operator", (None, "title", "body", "title_body")
        )
    rank_profile_2 = st.sidebar.selectbox(
        "Ranking 2: rank profile", RANK_PROFILE_OPTIONS
    )
    grammar_operator_2 = st.sidebar.selectbox("Ranking 2: Grammar", ("AND", "OR"))
    ann_operator_2 = None
    if RANK_PROFILE_EMBEDDING[RANK_PROFILE_MAP[rank_profile_2]] in AVAILABLE_EMBEDDINGS:
        ann_operator_2 = st.sidebar.selectbox(
            "Ranking 2: ANN operator", (None, "title", "body", "title_body")
        )
    number_queries = int(st.text_input("Number of queries to send", "20"))

    hits = int(st.text_input("Number of hits to evaluate per query", "10"))

    if st.button("Evaluate"):
        query_relevance = sample_query_relevance_data(number_queries=number_queries)

        model1 = retrieve_model(
            RANK_PROFILE_EMBEDDING[RANK_PROFILE_MAP[rank_profile_1]]
        )
        records_1, aggregate_metrics_1, position_freq_1 = evaluate(
            query_relevance=query_relevance,
            rank_profile=RANK_PROFILE_MAP[rank_profile_1],
            grammar_any=GRAMMAR_OPERATOR_MAP[grammar_operator_1],
            vespa_url=vespa_url,
            vespa_port=vespa_port,
            hits=int(hits),
            model=model1,
            ann=ann_operator_1,
        )

        model2 = retrieve_model(
            RANK_PROFILE_EMBEDDING[RANK_PROFILE_MAP[rank_profile_2]]
        )
        records_2, aggregate_metrics_2, position_freq_2 = evaluate(
            query_relevance=query_relevance,
            rank_profile=RANK_PROFILE_MAP[rank_profile_2],
            grammar_any=GRAMMAR_OPERATOR_MAP[grammar_operator_2],
            vespa_url=vespa_url,
            vespa_port=vespa_port,
            hits=int(hits),
            model=model2,
            ann=ann_operator_2,
        )
        position_freqs = [position_freq_1, position_freq_2]
        ranking_names = [
            aggregate_metrics_1["rank_name"],
            aggregate_metrics_2["rank_name"],
        ]
        results_summary = [aggregate_metrics_1, aggregate_metrics_2]

        display_results(position_freqs, ranking_names, results_summary, hits)


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

    rank_profile = st.selectbox("Select desired rank profile", RANK_PROFILE_OPTIONS)
    rank_profile = RANK_PROFILE_MAP[rank_profile]

    grammar_operator = st.selectbox(
        "Which grammar operator to apply fo query tokens", ("AND", "OR")
    )

    embedding = None
    ann_operator = None
    if RANK_PROFILE_EMBEDDING[rank_profile] in AVAILABLE_EMBEDDINGS:
        ann_operator = st.selectbox(
            "ANN operator", (None, "title", "body", "title_body")
        )
        model = retrieve_model(RANK_PROFILE_EMBEDDING[rank_profile])
        embedding = create_document_embedding(
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
            rank_profile=rank_profile,
            grammar_any=GRAMMAR_OPERATOR_MAP[grammar_operator],
            embedding=embedding,
            tracelevel=tracelevel,
            ann=ann_operator,
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


def parse_vespa_json(data):
    ranking = []
    if "children" in data["root"]:
        ranking = [
            (hit["fields"]["id"], hit["relevance"])
            for hit in data["root"]["children"]
            if "fields" in hit
        ]
    return (
        ranking,
        data["root"]["fields"]["totalCount"] / data["root"]["coverage"]["documents"],
    )


# @st.cache(persist=True)
def evaluate(
    query_relevance,
    rank_profile,
    grammar_any,
    vespa_url,
    vespa_port,
    hits,
    model=None,
    ann=None,
):
    if grammar_any:
        grammar_name = "OR"
    else:
        grammar_name = "AND"

    rank_name = rank_profile + ", " + grammar_name
    if ann:
        rank_name += ", ANN {}".format(ann)

    number_queries = 0
    total_rr = 0
    total_count = 0
    start_time = time()
    records = []
    position_count = [0] * min(hits, LIMIT_HITS_GRAPH)
    matched_ratio_sum = 0
    for qid, (query, relevant_id) in query_relevance.items():
        rr = 0
        embedding = None
        if model is not None:
            embedding = create_document_embedding(
                text=query,
                model=model["model"],
                model_source=model["model_source"],
                normalize=True,
            )
        request_body = create_vespa_body_request(
            query=query,
            rank_profile=rank_profile,
            grammar_any=grammar_any,
            hits=hits,
            offset=0,
            summary="minimal",
            embedding=embedding,
            ann=ann,
        )
        vespa_result = vespa_search(
            vespa_url=vespa_url, vespa_port=vespa_port, body=request_body
        )
        ranking, matched_ratio = parse_vespa_json(data=vespa_result)
        matched_ratio_sum += matched_ratio
        count = 0
        for rank, hit in enumerate(ranking):
            if hit[0] == relevant_id:
                rr = 1 / (rank + 1)
                if rank < LIMIT_HITS_GRAPH:
                    position_count[rank] += 1
                count += 1
        records.append({"qid": qid, "rr": rr})
        total_count += count
        total_rr += rr
        number_queries += 1
    execution_time = time() - start_time
    aggregate_metrics = {
        "rank_name": rank_name,
        "number_queries": number_queries,
        "qps": number_queries / execution_time,
        "mrr": total_rr / number_queries,
        "recall": total_count / number_queries,
        "average_matched": matched_ratio_sum / number_queries,
    }
    position_freq = [count / number_queries for count in position_count]
    return records, aggregate_metrics, position_freq


def create_vespa_body_request(
    query,
    rank_profile,
    grammar_any=False,
    hits=10,
    offset=0,
    summary=None,
    embedding=None,
    tracelevel=None,
    ann=None,
):
    yql = "select * from sources * where "

    if grammar_any:
        yql = yql + '([{"grammar": "any"}]userInput(@userQuery))'
    else:
        yql = yql + "(userInput(@userQuery))"

    body = {
        "userQuery": query,
        "hits": hits,
        "offset": offset,
        "ranking": {"profile": rank_profile, "listFeatures": "true"},
        "timeout": 1,
        "presentation.format": "json",
    }
    if tracelevel:
        body.update({"tracelevel": tracelevel})
    if summary == "minimal":
        body.update({"summary": "minimal"})
    if embedding:
        if RANK_PROFILE_EMBEDDING[rank_profile] == "word2vec":
            body.update({"ranking.features.query(tensor)": str(embedding)})
            if ann == "title":
                yql = (
                    yql
                    + ' or ([{{"targetNumHits": 1000, "label": "nns"}}]nearestNeighbor({},{}))'.format(
                        "title_word2vec", "tensor"
                    )
                )
            elif ann == "body":
                yql = (
                    yql
                    + ' or ([{{"targetNumHits": 1000, "label": "nns"}}]nearestNeighbor({},{}))'.format(
                        "body_word2vec", "tensor"
                    )
                )
            elif ann == "title_body":
                yql = (
                    yql
                    + ' or ([{{"targetNumHits": 1000, "label": "nns"}}]nearestNeighbor({},{})) or ([{{"targetNumHits": 1000, "label": "nns"}}]nearestNeighbor({},{}))'.format(
                        "title_word2vec", "tensor", "body_word2vec", "tensor"
                    )
                )
        elif RANK_PROFILE_EMBEDDING[rank_profile] == "gse":
            body.update({"ranking.features.query(tensor_gse)": str(embedding)})
            if ann == "title":
                yql = (
                    yql
                    + ' or ([{{"targetNumHits": 1000, "label": "nns"}}]nearestNeighbor({},{}))'.format(
                        "title_gse", "tensor_gse"
                    )
                )
            elif ann == "body":
                yql = (
                    yql
                    + ' or ([{{"targetNumHits": 1000, "label": "nns"}}]nearestNeighbor({},{}))'.format(
                        "body_gse", "tensor_gse"
                    )
                )
            elif ann == "title_body":
                yql = (
                    yql
                    + ' or ([{{"targetNumHits": 1000, "label": "nns"}}]nearestNeighbor({},{})) or ([{{"targetNumHits": 1000, "label": "nns"}}]nearestNeighbor({},{}))'.format(
                        "title_gse", "tensor_gse", "body_gse", "tensor_gse"
                    )
                )
        elif RANK_PROFILE_EMBEDDING[rank_profile] == "bert":
            body.update({"ranking.features.query(tensor_bert)": str(embedding)})
            if ann == "title":
                yql = (
                    yql
                    + ' or ([{{"targetNumHits": 1000, "label": "nns"}}]nearestNeighbor({},{}))'.format(
                        "title_bert", "tensor_bert"
                    )
                )
            elif ann == "body":
                yql = (
                    yql
                    + ' or ([{{"targetNumHits": 1000, "label": "nns"}}]nearestNeighbor({},{}))'.format(
                        "body_bert", "tensor_bert"
                    )
                )
            elif ann == "title_body":
                yql = (
                    yql
                    + ' or ([{{"targetNumHits": 1000, "label": "nns"}}]nearestNeighbor({},{})) or ([{{"targetNumHits": 1000, "label": "nns"}}]nearestNeighbor({},{}))'.format(
                        "title_bert", "tensor_bert", "body_bert", "tensor_bert"
                    )
                )
        else:
            raise NotImplementedError

    yql = yql + ";"
    body.update({"yql": yql})
    return body


def vespa_search(vespa_url, vespa_port, body):

    r = post(vespa_url + ":" + vespa_port + "/search/", json=body)
    return r.json()


if __name__ == "__main__":
    main()
