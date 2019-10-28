#!/usr/bin/env python3 
import json
import sys
from nltk.tokenize import sent_tokenize
import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece
import numpy as np 
 
def generate_vespa_feed(data,queries,passage_id):
  query = data['query']
  query_id = int(data['query_id'])
  is_selected = 0 
  for passage in data['passages']:
    is_selected = is_selected + passage['is_selected']
  if is_selected == 0:
    return []
  queries.append((query_id,query,is_selected))
  documents = []
  for passage in data['passages']:
    passage_text = passage['passage_text']
    questions = []
    if passage["is_selected"] == 1:
      questions.append(query_id)
      
    sentences = sent_tokenize(passage_text)
    context = [passage_text for s in sentences]
    sentence_embeddings = session.run(response_embeddings,feed_dict={answer:sentences,answer_context:context})['outputs']
    for i in range(0,len(sentences)):
      vespa_sentence_doc = {
        "put": "id:msmarco:sentence::%s" % str(passage_id) + "-" + str(i),
        "fields": {
          "questions": questions,
          "context_id": passage_id,
          "text": sentences[i],
          "dataset": "msmarco",
          "sentence_embedding": {
            "values" : sentence_embeddings[i].tolist()
          }
        }
      }
      documents.append(vespa_sentence_doc)
    vespa_context_doc = {
      "put":"id:msmarco:context::%i" % passage_id,
      "fields": {
        "text": passage_text,
        "dataset": "msmarco",
        "context_id": passage_id,
        "questions": questions
      } 
    }     
    documents.append(vespa_context_doc)
    passage_id +=1
  return documents
    
print("Downloading QA sentence encoder - about 1GB which needs to be downloaded")
g = tf.Graph()
with g.as_default():
  module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/1")
  answer = tf.compat.v1.placeholder(tf.string)
  answer_context = tf.compat.v1.placeholder(tf.string)
  response_embeddings = module(
    dict(input=answer,
         context=answer_context),
    signature="response_encoder", as_dict=True)
  
  question_input = tf.compat.v1.placeholder(tf.string) 
  question_embeddings = module(
    dict(input=question_input),
    signature="question_encoder", as_dict=True)

  init_op = tf.group([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
g.finalize()
         
# Initialize session.
session = tf.compat.v1.Session(graph=g)
session.run(init_op)
print("Done creating TF session")

query_file = open("msmarco_queries.txt","w")
feed_file = open("msmarco_vespa_feed.json","w")

queries = []
passage_id=0
for line in sys.stdin:
  line = line.strip()
  j  = json.loads(line)
  docs = generate_vespa_feed(j, queries, passage_id)
  passage_id +=1
  for d in docs:
    json.dump(d,feed_file)
    feed_file.write("\n")

def chunks(l, n):
  for i in range(0, len(l), n):
    yield l[i:i + n]
   
for chunk in chunks(queries,200): 
  chunk_queries = [str(q[1]) for q in chunk]
  embeddings = session.run(question_embeddings,feed_dict={question_input:chunk_queries})['outputs']
  for i in range(0,len(chunk)):
    question_id,question,number_answers = chunk[i]
    query_file.write("%i\t%s\t%i\t%s\n" % (int(question_id),str(question),int(number_answers),str(embeddings[i].tolist())))

 
query_file.close()
feed_file.close() 
