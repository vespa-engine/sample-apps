#!/usr/bin/env python3 
import json
import sys
from nltk.tokenize import sent_tokenize
import tensorflow as tf
import tensorflow_hub as hub


def tensor2json(sentence_embeddings):
  sentences = {}
  sentence_id = 0
  for s in sentence_embeddings:
    x = 0
    sentences[sentence_id] = s.tolist()
    sentence_id = sentence_id + 1
  return sentences 
  

def generate_vespa_feed(data,queryfile):
  query = data['query']
  query_type  = data['query_type']
  query_id = data['query_id']
  is_selected = 0
  #Check if this at least one relevant passage is found
  for passage in data['passages']:
    is_selected = is_selected + passage['is_selected']
  if is_selected == 0:
    return []
  queryfile.write("%i\t%s\t%s\n" % (query_id,query,query_type))
  documents = []
  for passage in data['passages']:

    passage_text = passage['passage_text']
    passage_embedding = session.run(embeddings,feed_dict={text_sentences:[passage_text]})[0].tolist()

    sentences = sent_tokenize(passage_text)
    sentence_embeddings = session.run(embeddings,feed_dict={text_sentences:sentences})
    sentence_embeddings = tensor2json(sentence_embeddings)
    vespa_fields= {
      "query": query,
      "query_id": query_id,
      "query_type": query_type,
      "is_selected": passage['is_selected'],
      "passage_text": passage_text,
      "sentences": sentences,
      "n_sentences": len(sentences),
      "sentence_embeddings": {
        "blocks": sentence_embeddings 
      },
      "passage_embedding": {
        "values" : passage_embedding 
      }
    }
    documents.append(vespa_fields)
  return documents 
     

print("Downloading universal sentence encoder - about 1GB which needs to be downloaded")
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
module = hub.Module(module_url)
text_sentences = tf.compat.v1.placeholder(tf.string)
embeddings = module(text_sentences)

print("Done Downloading universal sentence encoder")

print("Starting TF session. Expect some warnings during initialization")
session = tf.compat.v1.Session() 
session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
 
print("Done starting TF session")

print("Starting to convert to Vespa json")
queryfile = open("queries.txt","w") 
feedfile = open("vespa_feed.json","w")
docid=0
for line in sys.stdin:
  line = line.strip()
  j  = json.loads(line)
  docs = generate_vespa_feed(j,queryfile)
  if len(docs) == 0:
    continue
  for d in docs:
    vespa_json = {
      "put": "id:msmarco:passage::%i" % docid,
      "fields": d
    }
    docid = docid+1
    json.dump(vespa_json,feedfile)
    feedfile.write("\n")
 
queryfile.close() 
feedfile.close() 
