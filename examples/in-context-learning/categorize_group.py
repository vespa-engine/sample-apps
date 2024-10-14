from datasets import load_dataset
import requests
from openai import OpenAI
import time


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' took {execution_time:.6f} seconds to execute.")
        return result
    return wrapper


# model_name = "gpt-4o"
model_name = "llama3.1"
# model_name = "llama3.2"
# model_name = "llama3.2:1b"
# client = OpenAI()
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)


# @timer_decorator
def get_label(text, examples):
    """
    Ask the LLM to categorize the input text into a single category

    :param text:        Input query
    :param examples:    Array of (text, relevance, label)
    :return:            String with predicted category name
    """
    examples.reverse()
    examples_string = "\n".join(
        [f"Text: {text}, Category: {predicted_label}" for text, relevance, predicted_label in examples])
    prompt = f"""
    Categorize the input text into a single category. 
    
    Text: {text}, Category:

    Think step by step and predict the category of the text, using examples provided below. 

    {examples_string}
    
    Do not provide any code in result. Provide just the category without providing any reasoning.
    Text: {text}, Category:
    """
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=20,
        messages=[
            {"role": "system", "content": "You are an assistant that categorize incoming questions to a bank."},
            {"role": "user", "content": prompt}
        ]
    )
    result = response.choices[0].message.content

    # Simple heuristics to extract the category, if possible
    # The LLM output is unpredictable - can also run a similarity search on output to predict category
    if result not in label_names:
        result = result.replace("Category: ", "")
        if result not in label_names:
            result = "UNKNOWN"
    return result


def parse_vespa_group_response(response):
    """
    Parse a Vespa Query into an array of (text, relevance, label) tuples
    :param response:    A JSON Query response in https://docs.vespa.ai/en/reference/default-result-format.html
    :return:            Array of examples in (text, relevance, label) tuples
    """
    result = []
    groups = response['root'].get('children', []).pop().get('children', []).pop().get('children', [])
    for group in groups:
        hits = group.get('children', [])
        for hit in hits.pop().get('children', []):
            label = hit['fields']['label']
            text = hit['fields']['text']
            relevance = hit['relevance']
            result.append((text, relevance, label))
    return result


def inference(text):
    """
    Fetch results from Vespa with different rank profiles.
    For simplicity, all queries here parse a result from a grouping expression,
    but queries can group on the label or not:

        To run without grouping the result on label (i.e. no result diversity), use
            all(group(1) max(1) each(max(10) each(output(summary()))))
        This means, all in _one_ group, with 10 hits total.

        To run _with_ grouping on the label, use
            all(group(label) max(10) each(max(2) each(output(summary()))))
        Here, two examples per label.

    {targetHits:10} for the NN operator should be large enough for the expected set of examples returned.

    Set the correct rank-profile in the ranking query parameter.
    The rank-profiles are configured in train.sd.

    timeout is set unusually high here, for laptop experiments.

    Note that the hybrid rank-profiles need
        userQuery() OR ({targetHits:10}nearestNeighbor ...)
    to generate rank scores from both operators.

    When experimenting with different embedders, make sure to set the name in _both_ the ranking profile and
    the query ranking feature like
        'input.query(query_embedding)': 'embed(arctic, @query)'
    Here, the "arctic" embedder model is used, also see services.xml configuration for configuring this.

    See start of file for how to configure the LLM client, opanAI/local / which local model to use.
    Remember to start ollama with correct model, too.

    :param text:    User query
    :return:        Array of examples in (text, relevance, label) tuples
    """

    query_request_bm25_group_20 = {
        'yql': 'select * from sources * where userQuery() limit 0 | all(group(label) max(10) each(max(2) each(output(summary()))))',
        'query': text,
        'timeout': '30s',
        'ranking': 'bm25'
    }
    query_request_bm25_no_group_10 = {
        'yql': 'select * from sources * where userQuery() limit 0 | all(group(1) max(1) each(max(10) each(output(summary()))))',
        'query': text,
        'timeout': '30s',
        'ranking': 'bm25'
    }
    query_request_hybrid_e5_no_group_10_normalized_bm25 = {
        'yql': 'select * from sources * where (userQuery() OR ({targetHits:10}nearestNeighbor(doc_embedding_e5, query_embedding) )) limit 0 | all(group(1) max(1) each(max(10) each(output(summary()))))',
        'query': text,
        'input.query(query_embedding)': 'embed(e5, @query)',
        'timeout': '30s',
        'ranking': 'hybrid_e5_normalized_bm25'
    }
    query_request_sim_e5_no_group_10 = {
        'yql': 'select * from sources * where ({targetHits:10}nearestNeighbor(doc_embedding_e5, query_embedding)) limit 0 | all(group(1) max(1) each(max(10) each(output(summary()))))',
        'query': text,
        'input.query(query_embedding)': 'embed(e5, @query)',
        'timeout': '30s',
        'ranking': 'sim_e5'
    }
    query_request_sim_arctic_no_group_10 = {
        'yql': 'select * from sources * where ({targetHits:10}nearestNeighbor(doc_embedding_arctic, query_embedding)) limit 0 | all(group(1) max(1) each(max(10) each(output(summary()))))',
        'query': text,
        'input.query(query_embedding)': 'embed(arctic, @query)',
        'timeout': '30s',
        'ranking': 'sim_arctic'
    }
    query_request_sim_e5_no_group_5 = {
        'yql': 'select * from sources * where ({targetHits:5}nearestNeighbor(doc_embedding_e5, query_embedding)) limit 0 | all(group(1) max(1) each(max(5) each(output(summary()))))',
        'query': text,
        'input.query(query_embedding)': 'embed(e5, @query)',
        'timeout': '30s',
        'ranking': 'sim_e5'
    }
    query_request_sim_e5_no_group_2 = {
        'yql': 'select * from sources * where ({targetHits:2}nearestNeighbor(doc_embedding_e5, query_embedding)) limit 0 | all(group(1) max(1) each(max(2) each(output(summary()))))',
        'query': text,
        'input.query(query_embedding)': 'embed(e5, @query)',
        'timeout': '30s',
        'ranking': 'sim_e5'
    }
    query_request_sim_e5_no_group_15 = {
        'yql': 'select * from sources * where ({targetHits:15}nearestNeighbor(doc_embedding_e5, query_embedding)) limit 0 | all(group(1) max(1) each(max(15) each(output(summary()))))',
        'query': text,
        'input.query(query_embedding)': 'embed(e5, @query)',
        'timeout': '30s',
        'ranking': 'sim_e5'
    }
    query_request_sim_e5_no_group_20 = {
        'yql': 'select * from sources * where ({targetHits:20}nearestNeighbor(doc_embedding_e5, query_embedding)) limit 0 | all(group(1) max(1) each(max(20) each(output(summary()))))',
        'query': text,
        'input.query(query_embedding)': 'embed(e5, @query)',
        'timeout': '30s',
        'ranking': 'sim_e5'
    }
    query_request_sim_e5_group_10 = {
        'yql': 'select * from sources * where ({targetHits:100}nearestNeighbor(doc_embedding_e5, query_embedding)) limit 0 | all(group(label) max(10) each(max(1) each(output(summary()))))',
        'query': text,
        'input.query(query_embedding)': 'embed(e5, @query)',
        'timeout': '30s',
        'ranking': 'sim_e5'
    }
    query_request_sim_e5_group_20 = {
        'yql': 'select * from sources * where ({targetHits:100}nearestNeighbor(doc_embedding_e5, query_embedding)) limit 0 | all(group(label) max(10) each(max(2) each(output(summary()))))',
        'query': text,
        'input.query(query_embedding)': 'embed(e5, @query)',
        'timeout': '30s',
        'ranking': 'sim_e5'
    }
    query_request_sim_e5_group_30 = {
        'yql': 'select * from sources * where ({targetHits:100}nearestNeighbor(doc_embedding_e5, query_embedding)) limit 0 | all(group(label) max(10) each(max(3) each(output(summary()))))',
        'query': text,
        'input.query(query_embedding)': 'embed(e5, @query)',
        'timeout': '30s',
        'ranking': 'sim_e5'
    }
    query_request = query_request_bm25_no_group_10
    response = requests.post(f"http://localhost:8080/search/", json=query_request)
    if response.ok:
        return parse_vespa_group_response(response.json())
    else:
        print("Search request failed with response " + str(response.json()))
        return []


def get_label_from_retrieval(candidates):
    """
    Return a (label, relevance) tuple from a candidate list.
    In current implementation, return the top scoring label (ordered by relevance)
    Alternative implementations include returning the most frequent label, possibly weighted by position

    :param candidates:  List of (text, relevance, label) tuples
    :return:            (label, relevance) tuple
    """
    return candidates[0][2], candidates[0][1]


ds = load_dataset("PolyAI/banking77", split="test")
labels = dict()
label_names = []
with open("labels-map.txt", "r") as f:
    for line in f:
        id, label_text = line.strip().split("\t")
        labels[int(id)] = label_text.strip()
        label_names.append(label_text)

n = 0
correct = 0
print(f"category\tsize\trelevance\tretrieved_label\tpredicted_label\tlabel_text\ttext")
for row in ds:
    text = row['text'].replace('\n', '')
    label = int(row['label'])
    label_text = labels[label]
    examples = inference(text)
    retrieved_label, relevance = get_label_from_retrieval(examples)
    predicted_label = get_label(text, examples)

    # Categories
    # 0: Retrieved and Predicted == Actual
    # 1: Retrieved and NOT Predicted == Actual
    # 2: NOT Retrieved but Predicted == Actual
    # 3: NONE == Actual

    if retrieved_label == label_text:
        if predicted_label == label_text:
            category = 0
        else:
            category = 1
    else:
        if predicted_label == label_text:
            category = 2
        else:
            category = 3

    print(f"{category}\t{len(examples)}\t{relevance}\t{retrieved_label}\t{predicted_label}\t{label_text}\t{text}")

    if predicted_label == label_text:
        correct += 1
    n += 1
    if n == 1000:
        break
print(f"Categorization Accuracy: {correct / n}")
