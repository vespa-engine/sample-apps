#! /usr/bin/env python3

import sys
import numpy as np
import tensorflow_hub as hub
import json


def create_document_embedding(text, model, normalize=True):
    vector = model([text]).numpy()
    if normalize:
        norm = np.linalg.norm(vector)
        if norm > 0.0:
            vector = vector / norm
    return vector.tolist()[0]


def create_vespa_update(
    doc_id, title_field_name, title_field_values, body_field_name, body_field_values
):
    return {
        "update": "id:msmarco:msmarco::{}".format(doc_id),
        "fields": {
            title_field_name: {"assign": {"values": title_field_values}},
            body_field_name: {"assign": {"values": body_field_values}},
        },
    }


def main(input_file_path, output_file_path, embedding_method):

    if embedding_method == "word2vec":
        print("Using word2vec model")
        model = hub.load("https://tfhub.dev/google/Wiki-words-500-with-normalization/2")
    elif embedding_method == "gse":
        print("Using universal sentence encoder model")
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    else:
        raise NotImplementedError

    with open(input_file_path, "r") as file_in:
        with open(output_file_path, "w") as file_out:
            count = 0
            for line in file_in:
                if line not in ["[\n", "]\n"]:
                    count += 1
                    vespa_doc = json.loads(line.rstrip(",\n"))
                    vespa_doc_id = vespa_doc["fields"]["id"]
                    vespa_doc_title = vespa_doc["fields"]["title"]
                    vespa_doc_body = vespa_doc["fields"]["body"]

                    print("{} - id: {}".format(count, vespa_doc_id))

                    title_vector = create_document_embedding(
                        vespa_doc_title, model=model, normalize=True
                    )
                    body_vector = create_document_embedding(
                        vespa_doc_body, model=model, normalize=True
                    )
                    data_to_update = create_vespa_update(
                        doc_id=vespa_doc_id,
                        title_field_name="title_word2vec",
                        title_field_values=title_vector,
                        body_field_name="body_word2vec",
                        body_field_values=body_vector,
                    )
                    file_out.write(json.dumps(data_to_update))
                    file_out.write("\n")


if __name__ == "__main__":
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    embedding_method = sys.argv[3] if len(sys.argv) > 3 else "word2vec"
    main(input_file_path, output_file_path, embedding_method)
