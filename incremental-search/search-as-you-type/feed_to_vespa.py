#!/usr/bin/env python3
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
import os
import subprocess
import sys
import yaml


def find(json, path, separator = "."):
    if len(path) == 0: return json
    head, _, rest = path.partition(separator)
    return find(json[head], rest) if head in json else None


# extract <id> from form id:open:doc::<id>
def get_document_id(id):
    return id[id.rfind(":")+1:]


def call(args):
    proc = subprocess.Popen(args, stdout=subprocess.PIPE)
    (out, err) = proc.communicate()
    return out

def vespa_get(endpoint, operation, options):
    endpoint = endpoint[:-1] if endpoint.endswith("/") else endpoint
    url = "{0}/{1}?{2}".format(endpoint, operation, "&".join(options))
    print(url)
    return call([
        "curl",
        "-gsS",
        url ])


def vespa_delete(endpoint, operation, options):
    endpoint = endpoint[:-1] if endpoint.endswith("/") else endpoint
    url = "{0}/{1}?{2}".format(endpoint, operation, "&".join(options))
    return call([
        "curl",
        "-gsS",
        "-X", "DELETE",
        url
    ])


def vespa_post(endpoint, doc, docid, namespace):
    endpoint = endpoint[:-1] if endpoint.endswith("/") else endpoint
    url = "{0}/document/v1/{1}/doc/docid/{2}".format(endpoint, namespace, docid)
    return call([
        "curl",
        "-sS",
        "-H", "Content-Type:application/json",
        "-X", "POST",
        "--data-binary", "{0}".format(doc),
        url
    ])


def vespa_visit(endpoint, namespace, continuation = None):
    options = []
    options.append("wantedDocumentCount=500")
    if continuation is not None and len(continuation) > 0:
        options.append("&continuation={0}".format(continuation))
    response = vespa_get(endpoint, "document/v1/{0}/doc/docid".format(namespace), options)
    try:
        return json.loads(response)
    except:
        print("Unable to parse JSON response from {0}. Should not happen, endpoint down? response: {1}".format(endpoint, response))
        sys.exit(1)
    return {}


def vespa_remove(endpoint, doc_ids, namespace):
    options = []
    for doc_id in doc_ids:
        id = get_document_id(doc_id)
        print("Removing: {0}".format(id))
        vespa_delete(endpoint, "document/v1/{0}/doc/docid/{1}".format(namespace, id), options)


def vespa_feed(endpoint, feed, namespace):
    for doc in get_docs(feed):
        document_id = find(doc, "fields.namespace") +  find(doc, "fields.path")
        print(vespa_post(endpoint, json.dumps(doc), document_id, namespace))


def get_docs(index):
    file = open(index, "r", encoding='utf-8')
    return json.load(file)


def get_indexed_docids(endpoint, namespace):
    docids = set()
    continuation = ""
    while continuation is not None:
        json = vespa_visit(endpoint, namespace, continuation)
        documents = find(json, "documents")
        if documents is not None:
            ids = [ find(document, "id") for document in documents ]
            for id in ids:
                print("Found {0}".format(id))
            docids.update(ids)
        continuation = find(json, "continuation")
    return docids


def get_feed_docids(feed, namespace):
    with open(feed, "r", encoding='utf-8') as f:
        feed_json = json.load(f)
    return set([ "id:{0}:doc::".format(namespace) + find(doc, "fields.namespace") + find(doc, "fields.path") for doc in feed_json ])


def print_header(msg):
    print("")
    print("*" * 80)
    print("* {0}".format(msg))
    print("*" * 80)


def read_config():
    with open("_config.yml", "r") as f:
        return yaml.safe_load(f)


def update_endpoint(endpoint, config):
    do_remove_index = config["search"]["do_index_removal_before_feed"]
    do_feed = config["search"]["do_feed"]
    namespace = config["search"]["namespace"]

    endpoint_url = endpoint["url"]
    endpoint_indexes = endpoint["indexes"]

    print_header("Retrieving already indexed document ids for endpoint {0}".format(endpoint_url))
    docids_in_index = get_indexed_docids(endpoint_url, namespace)
    print("{0} documents found.".format(len(docids_in_index)))

    if do_remove_index:
        print_header("Removing all indexed documents in {0}".format(endpoint_url))
        vespa_remove(endpoint_url, docids_in_index, namespace)
        print("{0} documents removed.".format(len(docids_in_index)))

    if do_feed:
        docids_in_feed = set()
        print_header("Parsing feed file(s) for document ids")
        for index in endpoint_indexes:
            assert os.path.exists(index)
            docids_in_feed = docids_in_feed.union(get_feed_docids(index, namespace))
        print("{0} documents found.".format(len(docids_in_feed)))

        if len(docids_in_feed) == 0:
            return

        docids_to_remove = docids_in_index.difference(docids_in_feed)
        if len(docids_to_remove) > 0:
            print_header("Removing indexed documents not in feed in {0}".format(endpoint_url))
            for id in docids_to_remove:
                print("To Remove: {0}".format(id))
            vespa_remove(endpoint_url, docids_to_remove, namespace)
            print("{0} documents removed.".format(len(docids_to_remove)))
        else:
            print("No documents to be removed.")

        for index in endpoint_indexes:
            print_header("Feeding {0} to {1}...".format(index, endpoint_url))
            print(vespa_feed(endpoint_url, index, namespace))

        print("{0} documents fed.".format(len(docids_in_feed)))


def main():
    config = read_config()
    for endpoint in config["search"]["feed_endpoints"]:
        update_endpoint(endpoint, config)


if __name__ == "__main__":
    main()
