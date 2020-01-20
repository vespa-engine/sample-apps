#! /usr/bin/env python3

import sys
import numpy as np
import tensorflow_hub as hub
import json

word2vec_embed = hub.load(
    "https://tfhub.dev/google/Wiki-words-500-with-normalization/2"
)


def create_document_embedding(text, model="word2vec", normalize=True):
    if model == "word2vec":
        vector = word2vec_embed([text]).numpy()
    else:
        raise NotImplementedError
    if normalize:
        vector = vector / np.linalg.norm(vector)
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


def main(input_file_path, output_file_path):
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

                    title_word2vec = create_document_embedding(
                        vespa_doc_title, model="word2vec", normalize=True
                    )
                    body_word2vec = create_document_embedding(
                        vespa_doc_body, model="word2vec", normalize=True
                    )
                    data_to_update = create_vespa_update(
                        doc_id=vespa_doc_id,
                        title_field_name="title_word2vec",
                        title_field_values=title_word2vec,
                        body_field_name="body_word2vec",
                        body_field_values=body_word2vec,
                    )
                    file_out.write(json.dumps(data_to_update))
                    file_out.write("\n")


if __name__ == "__main__":
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    main(input_file_path, output_file_path)
