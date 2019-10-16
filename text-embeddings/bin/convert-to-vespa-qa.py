#!/usr/bin/env python3 
import json
import sys
from nltk.tokenize import sent_tokenize
import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece

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
    answer_tensor = session.run(response_embeddings,feed_dict={answer:[passage_text],answer_context:[passage_text]})['outputs'][0]
    sentences = sent_tokenize(passage_text)
    sentence_embeddings = []
    for sentence in sentences:
      answer_tensor = session.run(response_embeddings,feed_dict={answer:[sentence],answer_context:[passage_text]})['outputs'][0]
      sentence_embeddings.append(answer_tensor)
    sentence_embeddings = tensor2json(sentence_embeddings)
#!/usr/bin/env python3 
import json
import sys
from nltk.tokenize import sent_tokenize
import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece

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
    answer_tensor = session.run(response_embeddings,feed_dict={answer:[passage_text],answer_context:[passage_text]})['outputs'][0]
    sentences = sent_tokenize(passage_text)
    sentence_embeddings = []
    for sentence in sentences:
      answer_tensor = session.run(response_embeddings,feed_dict={answer:[sentence],answer_context:[passage_text]})['outputs'][0]
      sentence_embeddings.append(answer_tensor)
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
        "values" : answer_tensor.tolist() 
      }
    }
    documents.append(vespa_fields)
  return documents 
     

print("Downloading universal sentence encoder - about 1GB which needs to be downloaded")
g = tf.Graph()
with g.as_default():
  module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/1")
  answer = tf.compat.v1.placeholder(tf.string) 
  answer_context = tf.compat.v1.placeholder(tf.string)
  response_embeddings = module(
    dict(input=answer,
         context=answer_context),
    signature="response_encoder", as_dict=True)
  
  init_op = tf.group([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
g.finalize()

# Initialize session.
session = tf.compat.v1.Session(graph=g)
session.run(init_op)
print("Done creating TF session")

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
        "values" : answer_tensor.tolist() 
      }
    }
    documents.append(vespa_fields)
  return documents 
     

print("Downloading universal sentence encoder - about 1GB which needs to be downloaded")
g = tf.Graph()
with g.as_default():
  module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/1")
  answer = tf.compat.v1.placeholder(tf.string) 
  answer_context = tf.compat.v1.placeholder(tf.string)
  response_embeddings = module(
    dict(input=answer,
         context=answer_context),
    signature="response_encoder", as_dict=True)
  
  init_op = tf.group([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
g.finalize()

# Initialize session.
session = tf.compat.v1.Session(graph=g)
session.run(init_op)
print("Done creating TF session")

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

