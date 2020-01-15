#! /usr/bin/env python3

import numpy as np
import tensorflow_hub as hub
import json

word2vec_embed = hub.load(
    "https://tfhub.dev/google/Wiki-words-500-with-normalization/2"
)


def create_document_embedding_word2vec(text, normalize=True):
    vector = word2vec_embed([text]).numpy()
    if normalize:
        vector = vector / np.linalg.norm(vector)
    return vector.tolist()[0]


def create_vespa_update(doc_id, title_word2vec, body_word2vec):
    return {
        "update": "id:msmarco:msmarco::{}".format(doc_id),
        "fields": {
            "title_word2vec": {"assign": {"values": title_word2vec}},
            "body_word2vec": {"assign": {"values": body_word2vec}},
        },
    }


def main():
    # todo: include file paths as script parameter
    # todo: add google sentence encoder
    # todo: do a test run with vespa feed to confirm the format is valid
    input_file_path = "../../data/delete_this.json"
    output_file_path = "../../data/vespa_embeddings.json"
    with open(input_file_path, "r") as file_in:
        with open(output_file_path, "w") as file_out:
            for line in file_in:
                if line not in ["[\n", "]\n"]:
                    vespa_doc = json.loads(line.rstrip(",\n"))
                    vespa_doc_id = vespa_doc["fields"]["id"]
                    vespa_doc_title = vespa_doc["fields"]["title"]
                    vespa_doc_body = vespa_doc["fields"]["body"]

                    title_word2vec = create_document_embedding_word2vec(vespa_doc_title)
                    body_word2vec = create_document_embedding_word2vec(vespa_doc_body)

                    data_to_update = create_vespa_update(
                        doc_id=vespa_doc_id,
                        title_word2vec=title_word2vec,
                        body_word2vec=body_word2vec,
                    )
                    file_out.write(json.dumps(data_to_update))
                    file_out.write("\n")


if __name__ == "__main__":
    main()
