#! /usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import sys
import json
import urllib.parse
import requests


def parse_embedding(hit_json):
    return hit_json["fields"]["embedding"]["values"]

def query_user_embedding(user_id):
    yql = 'select * from sources user where user_id contains "{}"'.format(user_id)
    url = 'http://localhost:8080/search/?yql={}&hits=1'.format(urllib.parse.quote_plus(yql))  
    result = requests.get(url).json()
    return parse_embedding(result["root"]["children"][0])

def query_news(user_vector, hits, filter):
    nn_annotations = [
        'targetHits:{}'.format(hits)
        ]
    nn_annotations = '{' + ','.join(nn_annotations) + '}'
    nn_search = '({}nearestNeighbor(embedding, user_embedding))'.format(nn_annotations)

    data = {
        'hits': hits,
        'yql': 'select * from sources news where {} {}'.format(nn_search, filter),
        'ranking.features.query(user_embedding)': str(user_vector),
        'ranking.profile': 'recommendation'
    }
    return requests.post('http://localhost:8080/search/', json=data).json()


def main():
    user_id = sys.argv[1]
    hits = sys.argv[2] if len(sys.argv) > 2 else 10
    filter = sys.argv[3] if len(sys.argv) > 3 else ""

    user_vector = query_user_embedding(user_id)
    result = query_news(user_vector, int(hits), filter)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

