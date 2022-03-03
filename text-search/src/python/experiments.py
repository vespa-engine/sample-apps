import re
from requests import post
from time import time
from embedding import create_document_embedding


def parse_vespa_json(data):
    ranking = []
    matched_ratio = 0
    if "children" in data["root"]:
        ranking = [
            (hit["fields"]["id"], hit["relevance"])
            for hit in data["root"]["children"]
            if "fields" in hit
        ]
        matched_ratio = (
            data["root"]["fields"]["totalCount"] / data["root"]["coverage"]["documents"]
        )
    return (ranking, matched_ratio)


# @st.cache(persist=True)
def evaluate(
    query_relevance,
    parsed_rank_profile,
    grammar_operator,
    ann_operator,
    embedding_type,
    vespa_url,
    vespa_port,
    hits,
    model=None,
    limit_position_count=10,
):
    rank_name = (
        str(parsed_rank_profile)
        + str(grammar_operator)
        + str(ann_operator)
        + str(embedding_type)
    )

    number_queries = 0
    total_rr = 0
    total_count = 0
    start_time = time()
    records = []
    position_count = [0] * min(hits, limit_position_count)
    matched_ratio_sum = 0
    for qid, (query, relevant_id) in query_relevance.items():
        rr = 0
        embedding_vector = None
        if model is not None:
            embedding_vector = create_document_embedding(
                text=query,
                model=model["model"],
                model_source=model["model_source"],
                normalize=True,
            )
        request_body = create_vespa_body_request(
            query=query,
            parsed_rank_profile=parsed_rank_profile,
            grammar_operator=grammar_operator,
            ann_operator=ann_operator,
            embedding_type=embedding_type,
            hits=hits,
            offset=0,
            summary="minimal",
            embedding_vector=embedding_vector,
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
                if rank < limit_position_count:
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


def create_weakAND_operator(query, target_hits=1000):

    query = re.sub(" +", " ", query)  # remove multiple spaces
    query_tokens = query.strip().split(" ")
    terms = ", ".join(['default contains "' + token + '"' for token in query_tokens])
    return '([{"targetNumHits": ' + str(target_hits) + "}]weakAnd(" + terms + "))"


def create_ANN_operator(ann_operator, embedding, target_hits=1000):

    ann_parameters = {
        "title": {
            "word2vec": ["title_word2vec", "tensor"],
            "gse": ["title_gse", "tensor_gse"],
            "bert": ["title_bert", "tensor_bert"],
        },
        "body": {
            "word2vec": ["body_word2vec", "tensor"],
            "gse": ["body_gse", "tensor_gse"],
            "bert": ["body_bert", "tensor_bert"],
        },
    }

    if ann_operator in ["title", "body"]:
        return '([{{"targetNumHits": {}, "label": "nns"}}]nearestNeighbor({}, {}))'.format(
            *([target_hits] + ann_parameters[ann_operator][embedding])
        )
    elif ann_operator == "title_body":
        return (
            '([{{"targetNumHits": {}, "label": "nns1"}}]nearestNeighbor({}, {})) or '
            '([{{"targetNumHits": {}, "label": "nns2"}}]nearestNeighbor({}, {}))'.format(
                *(
                    [target_hits]
                    + ann_parameters["title"][embedding]
                    + [target_hits]
                    + ann_parameters["body"][embedding]
                )
            )
        )
    elif ann_operator is None:
        return None
    else:
        raise ValueError("Invalid ann_operator: {}".format(ann_operator))


def create_grammar_operator(query, grammar_operator):
    if grammar_operator == "OR":
        return '([{"grammar": "any"}]userInput(@userQuery))'
    elif grammar_operator == "AND":
        return "(userInput(@userQuery))"
    elif grammar_operator == "weakAND":
        return create_weakAND_operator(query)
    elif grammar_operator is None:
        return None
    elif grammar_operator is not None:
        raise ValueError("Invalid grammar operator {}.".format(grammar_operator))


def create_yql(query, grammar_operator, ann_operator, embedding):

    operators = []
    #
    # Parse grammar operator
    #
    parsed_grammar_operator = create_grammar_operator(query, grammar_operator)
    if parsed_grammar_operator is not None:
        operators.append(parsed_grammar_operator)
    #
    # Parse ANN operator
    #
    parsed_ann_operator = create_ANN_operator(ann_operator, embedding)
    if parsed_ann_operator is not None:
        operators.append(parsed_ann_operator)

    if not operators:
        raise ValueError("Choose at least one match phase operator.")

    yql = "select * from sources * where {}".format(" or ".join(operators))

    return yql


def create_vespa_body_request(
    query,
    parsed_rank_profile,
    grammar_operator,
    ann_operator,
    embedding_type,
    hits=10,
    offset=0,
    summary=None,
    embedding_vector=None,
    tracelevel=None,
):

    body = {
        "yql": create_yql(query, grammar_operator, ann_operator, embedding_type),
        "userQuery": query,
        "hits": hits,
        "offset": offset,
        "ranking": {"profile": parsed_rank_profile, "listFeatures": "true"},
        "timeout": 1,
        "presentation.format": "json",
    }
    if tracelevel:
        body.update({"tracelevel": tracelevel})
    if summary == "minimal":
        body.update({"summary": "minimal"})
    if embedding_vector:
        if embedding_type == "word2vec":
            body.update({"ranking.features.query(tensor)": str(embedding_vector)})
        elif embedding_type == "gse":
            body.update({"ranking.features.query(tensor_gse)": str(embedding_vector)})
        elif embedding_type == "bert":
            body.update({"ranking.features.query(tensor_bert)": str(embedding_vector)})
        else:
            raise NotImplementedError

    return body


def vespa_search(vespa_url, vespa_port, body):

    r = post(vespa_url + ":" + vespa_port + "/search/", json=body)
    return r.json()
