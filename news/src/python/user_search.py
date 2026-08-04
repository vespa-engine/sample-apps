#! /usr/bin/env python3
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import sys
import re
import json
import urllib.parse
import requests
import subprocess


def vespa_endpoint():
    result = subprocess.run(["vespa", "status", "--format", "plain"],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()

def vespa_cert():
    """Return (cert, key) paths from 'vespa curl', or None when no mTLS is configured."""
    result = subprocess.run(["vespa", "curl", "-n", "/"],
                            capture_output=True, text=True)
    output = result.stdout + result.stderr
    paths = dict(re.findall(r'--(cert|key) (\S+)', output))
    if "cert" in paths and "key" in paths:
        return (paths["cert"], paths["key"])
    return None

def parse_embedding(hit_json):
    return hit_json["fields"]["embedding"]["values"]

def query_user_embedding(user_id, url, cert):
    yql = 'select * from sources user where user_id contains "{}"'.format(user_id)
    url = '{}/search/?yql={}&hits=1'.format(url, urllib.parse.quote_plus(yql))
    result = requests.get(url, cert=cert).json()
    return parse_embedding(result["root"]["children"][0])

def query_news(user_vector, hits, filter, url, cert):
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
    return requests.post(f'{url}/search/', json=data, cert=cert).json()


def main():
    user_id = sys.argv[1]
    hits = sys.argv[2] if len(sys.argv) > 2 else 10
    filter = sys.argv[3] if len(sys.argv) > 3 else ""

    url = vespa_endpoint()
    cert = vespa_cert()

    user_vector = query_user_embedding(user_id, url, cert)
    result = query_news(user_vector, int(hits), filter, url, cert)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

