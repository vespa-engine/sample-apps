#! /usr/bin/env python3

from embedding import compute_and_send_image_embeddings
from vespa.application import Vespa

app = Vespa(url="http://localhost", port=8080)
compute_and_send_image_embeddings(app=app, batch_size=128, clip_model_names=['ViT-B/32'], schema="image_search")
