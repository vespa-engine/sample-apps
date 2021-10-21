#!/usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

from transformers import AutoTokenizer
from transformers import AutoModelWithLMHead
from sentence_transformers import SentenceTransformer 
from sentence_transformers import models 
import sys

PATH=sys.argv[1]
print('Saving model to %s' % PATH)

tokenizer = AutoTokenizer.from_pretrained("gsarti/scibert-nli")
model = AutoModelWithLMHead.from_pretrained("gsarti/scibert-nli")
model.save_pretrained(PATH)
tokenizer.save_pretrained(PATH)

embedding = models.BERT(PATH,max_seq_length=128,do_lower_case=True)
pooling_model = models.Pooling(embedding.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[embedding, pooling_model])
model.save(PATH)

