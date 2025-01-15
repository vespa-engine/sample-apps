#!/usr/bin/env python3
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
import os
import re
import subprocess
import sys
import yaml
import requests
from requests.adapters import HTTPAdapter, Retry
import urllib.parse

def find(json, path, separator = "."):
    if len(path) == 0: return json
    head, _, rest = path.partition(separator)
    return find(json[head], rest) if head in json else None


# extract <id> from form id:open:doc::<id>
def get_document_id(id):
    return id[id.rfind(":")+1:]


def get_private_key_path():
    private_key_path = "data-plane-private-key.pem"
    if not os.path.isfile(private_key_path):
        private_key_raw = os.environ['DATA_PLANE_PRIVATE_KEY']
        private_key = private_key_raw.replace(" ", "\n")
        with open(private_key_path, "w") as f:
            f.write("-----BEGIN PRIVATE KEY-----\n" + private_key  + "\n-----END PRIVATE KEY-----")
    return private_key_path


def get_public_cert_path():
    public_cert_path = "data-plane-public-key.pem"
    if not os.path.isfile(public_cert_path):
        public_cert_raw = os.environ['DATA_PLANE_PUBLIC_KEY']
        public_cert = public_cert_raw.replace(" ", "\n")
        with open(public_cert_path, "w") as f:
            f.write("-----BEGIN CERTIFICATE-----\n" + public_cert  + "\n-----END CERTIFICATE-----")
    return public_cert_path


def vespa_get(endpoint, operation, options):
    url = "{0}/{1}?{2}".format(endpoint, operation, "&".join(options))
    return session.get(url).json()


def vespa_delete(endpoint, operation, options):
    url = "{0}/{1}?{2}".format(endpoint, operation, "&".join(options))
    return session.delete(url).json()


def vespa_visit(endpoint, namespace, doc_type, continuation = None):
    options = []
    options.append("wantedDocumentCount=500")
    options.append("timeout=60s")
    if continuation is not None and len(continuation) > 0:
        options.append("&continuation={0}".format(continuation))
    return vespa_get(endpoint, "document/v1/{0}/{1}/docid".format(namespace,doc_type), options)


def vespa_remove(endpoint, doc_ids, namespace, doc_type):
    options = []
    for doc_id in doc_ids:
        id = get_document_id(doc_id)
        vespa_delete(endpoint, "document/v1/{0}/{1}/docid/{2}".format(namespace, doc_type, id), options)


def vespa_feed(endpoint, feed, namespace, doc_type):
    if doc_type not in ["paragraph", "term", "doc"]:
        raise ValueError(":error:Unknown vespa doc_type: {0}".format(doc_type))

    splits = re.split(r'/|\.', endpoint)
    app_string = splits[3] + '.' + splits[2]
    print("Feeding to app: {0} , endpoint: {1}".format(app_string, endpoint))

    process = subprocess.run(['vespa', 'feed', '-a', app_string, '-t', endpoint, feed], capture_output=True)

    # Print sderr if not empty
    if process.stderr:
        print("::group::VespaCLI-Error")
        print("::error::Errors reported by VespaCLI:")
        print(process.stderr.decode('utf-8'))
        print("::endgroup::")

    if process.returncode != 0:
        print("::error::Errors encountered while feeding Vespa application.")
        sys.exit(process.returncode)

    return process.stdout.decode('utf-8')


def get_docs(index):
    file = open(index, "r", encoding='utf-8')
    return json.load(file)


def get_indexed_docids(endpoint, namespace, doc_type):
    docids = set()
    continuation = ""
    while continuation is not None:
        json = vespa_visit(endpoint, namespace, doc_type, continuation)
        documents = find(json, "documents")
        if documents is not None:
            ids = [ find(document, "id") for document in documents ]
            for id in ids:
                # The document id might contain chars that needs to be escaped for the delete/put operation to work
                # also for comparison with what is in the feed
                docid = get_document_id(id) # return the last part
                encoded = urllib.parse.quote(docid) #escape
                id = id.replace(docid, encoded)
                docids.add(id)
        continuation = find(json, "continuation")
    return docids


def get_feed_docids(feed, namespace, doc_type):
    with open(feed, "r", encoding='utf-8') as f:
        feed_json = json.load(f)
    if doc_type == "doc":
        return set(["id:{0}:doc::".format(namespace) + find(doc, "fields.namespace") + find(doc, "fields.path") for doc in feed_json])
    elif doc_type == "term":
        return set(["id:{0}:term::".format(namespace) + str(find(doc, "fields.hash")) for doc in feed_json])
    elif doc_type == "paragraph":
        return set([doc['put'] for doc in feed_json])


def print_header(msg):
    print("")
    print("*" * 80)
    print("* {0}".format(msg))
    print("*" * 80)


def read_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def update_endpoint(endpoint, config):
    do_remove_index = config["search"]["do_index_removal_before_feed"]
    do_feed = config["search"]["do_feed"]
    namespace = config["search"]["namespace"]
    doc_type = config["search"]["doc_type"]

    endpoint_url = endpoint["url"]
    endpoint_url = endpoint_url[:-1] if endpoint_url.endswith("/") else endpoint_url
    endpoint_indexes = endpoint["indexes"]

    print_header("Retrieving already indexed document ids for endpoint {0}".format(endpoint_url))
    docids_in_index = get_indexed_docids(endpoint_url, namespace, doc_type)
    print("{0} documents found.".format(len(docids_in_index)))

    if do_remove_index:
        print_header("Removing all indexed documents in {0}".format(endpoint_url))
        vespa_remove(endpoint_url, docids_in_index, namespace, doc_type)
        print("{0} documents removed.".format(len(docids_in_index)))

    if do_feed:
        docids_in_feed = set()
        print_header("Parsing feed file(s) for document ids")
        for index in endpoint_indexes:
            assert os.path.exists(index)
            docids_in_feed = docids_in_feed.union(get_feed_docids(index, namespace, doc_type))
        print("{0} documents found.".format(len(docids_in_feed)))

        if len(docids_in_feed) == 0:
            return

        docids_to_remove = docids_in_index.difference(docids_in_feed)
        if len(docids_to_remove) > 0:
            print_header("Removing indexed documents not in feed in {0}".format(endpoint_url))
            for id in docids_to_remove:
                print("To Remove: {0}".format(id))
            vespa_remove(endpoint_url, docids_to_remove, namespace, doc_type)
            print("{0} documents removed.".format(len(docids_to_remove)))
        else:
            print("No documents to be removed.")

        for index in endpoint_indexes:
            print_header("Feeding {0} to {1}...".format(index, endpoint_url))
            print(vespa_feed(endpoint_url, index, namespace, doc_type))

        print("{0} documents fed.".format(len(docids_in_feed)))


def main():
    configuration_file = sys.argv[1]
    config = read_config(configuration_file)
    global session
    session = requests.Session()
    retries = Retry(total=10, connect=10,
        backoff_factor=0.8,
        status_forcelist=[ 500, 503, 504, 429 ]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.cert = (get_public_cert_path(), get_private_key_path())
    for endpoint in config["search"]["feed_endpoints"]:
        update_endpoint(endpoint, config)


if __name__ == "__main__":
    main()
