import unittest
import numpy as np
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer

from embedding import create_document_embedding


class TestEmbeddings(unittest.TestCase):
    def setUp(self) -> None:
        self.tf_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.sentence_bert = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    def test_create_document_embedding_tf_hub(self):
        vector = create_document_embedding(
            "this is a text", model=self.tf_model, model_source="tf_hub", normalize=True
        )
        print(vector)
        print(type(vector))
        print(len(vector))
        print(np.linalg.norm(vector))

    def test_create_document_embedding_bert(self):
        vector = create_document_embedding(
            "this is a text", model=self.sentence_bert, model_source="bert", normalize=True
        )
        print(vector)
        print(type(vector))
        print(len(vector))
        print(np.linalg.norm(vector))
