# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
from datasets import load_dataset
import json
import unicodedata

def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1536, # chars, not llm tokens
    chunk_overlap  = 0,
    length_function = len,
    is_separator_regex = False,
)

documents = load_dataset('Shitao/MLDR', "corpus-en", split='corpus', trust_remote_code=True)
feed_file = "/tmp/vespa_feed_file_en.json"
with open(feed_file, "w") as f:
    for doc in documents:
        id = doc["docid"]
        text = doc['text']
        chunks = text_splitter.create_documents([text])
        text_chunks = [chunk.page_content for chunk in chunks]
        text_chunks = [remove_control_characters(chunk) for chunk in text_chunks]
        vespa_feed_doc = {
            "put": "id:%s:doc::%s" %  ("en", id),
                "fields": {
                    "text": text_chunks
                }
            }
        f.write(json.dumps(vespa_feed_doc))
        f.write("\n")
