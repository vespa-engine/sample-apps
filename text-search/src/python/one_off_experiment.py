from exploration_app import sample_query_relevance_data, retrieve_model
from experiments import evaluate

query_relevance = sample_query_relevance_data(number_queries=None)

# {'rank_name': 'listwise_bm25_bert_title_body_allweakANDtitle_bodybert',
#  'number_queries': 500,
#  'qps': 0.5702334703000022,
#  'mrr': 0.6352814975343447,
#  'recall': 0.932,
#  'average_matched': 0.1837245548670878}
evaluation = evaluate(
    query_relevance=query_relevance,
    parsed_rank_profile="listwise_bm25_bert_title_body_all",
    grammar_operator="weakAND",
    ann_operator="title_body",
    embedding_type="bert",
    vespa_url="http://localhost",
    vespa_port="8080",
    hits=100,
    model=retrieve_model("bert"),
)

# {'rank_name': 'listwise_bm25_bert_title_body_allweakANDtitle_bodybert',
#  'number_queries': 500,
#  'qps': 1.5831257984703608,
#  'mrr': 0.7063329532120562,
#  'recall': 0.942,
#  'average_matched': 0.1837245548670878}
evaluation = evaluate(
    query_relevance=query_relevance,
    parsed_rank_profile="listwise_bm25_bert_title_body_all",
    grammar_operator="weakAND",
    ann_operator="title_body",
    embedding_type="bert",
    vespa_url="http://localhost",
    vespa_port="8080",
    hits=100,
    model=retrieve_model("bert"),
)


# {'rank_name': 'bm25_bert_title_body_allweakANDtitle_bodybert',
# 'number_queries': 500,
# 'qps': 0.5577734656647746,
# 'mrr': 0.6991934308152978,
# 'recall': 0.944,
# 'average_matched': 0.1837245548670878}
evaluation = evaluate(
    query_relevance=query_relevance,
    parsed_rank_profile="bm25_bert_title_body_all",
    grammar_operator="weakAND",
    ann_operator="title_body",
    embedding_type="bert",
    vespa_url="http://localhost",
    vespa_port="8080",
    hits=100,
    model=retrieve_model("bert"),
)
