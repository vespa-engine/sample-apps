#!/usr/bin/env python3 
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

from retrievalqaeval.sb_sed import infer_sentence_breaks
import json
import sys
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

def get_questions_to_answers(qas,sentence_breaks,context):
  questions_to_answers = []
  for qa in qas:
    question = qa["question"]
    answer_sentences = set()
    for answer in qa["answers"]:
      answer_start = answer["answer_start"]
      sentence = None
      for start, end in sentence_breaks:
        if start <= answer_start < end:
          sentence = context[start:end] #The sentence which the answer was found in
          break
      if sentence not in answer_sentences:
        answer_sentences.add(str(sentence))
    questions_to_answers.append((question,answer_sentences))
  return questions_to_answers

def make_vespa_feed_paragraph(questions,text,context_id):
  vespa_doc = {
    "put": "id:squad:context::%i" % context_id,
      "fields": {
        "text": text,
        "dataset": "squad",
        "questions": questions,
        "context_id": context_id,
      }
  }
  return vespa_doc

def make_vespa_feed(sentence_id,questions,sentence,sentence_embedding,context_id):
  vespa_doc = {
    "put": "id:squad:sentence::%i" % sentence_id,
      "fields": {
        "text": sentence,
        "dataset": "squad",
        "questions": questions,
        "context_id": context_id,
        "sentence_embedding": {
          "values": sentence_embedding.numpy().tolist()
        }
      }
  }
  return vespa_doc


print("Downloading QA universal sentence encoder - about 1GB which needs to be downloaded")
module = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3')

query_file = open("squad_queries.txt","w")
feed_file = open("squad_vespa_feed.json","w")
queries = []
with open(sys.argv[1]) as fp:
  question_id = 0
  sentence_id = 0
  data = json.load(fp)
 
  context_id = 0 
  for passage in data["data"]:
    for paragraph in passage["paragraphs"]:
      context = paragraph["context"]
      paragraph_questions = []#set of questions answered by sentences in paragraph
      sentence_breaks = list(infer_sentence_breaks(context))
      sentences = set([context[start:end] for (start,end) in sentence_breaks])
      questions_to_answers = get_questions_to_answers(paragraph['qas'],sentence_breaks,context)

      answer_sentences = {}
      for question,answers in questions_to_answers:
        queries.append((question_id,question,len(answers)))
        for a in answers: 
          if a in answer_sentences:
            answer_sentences[a].append(question_id)
          else:
            answer_sentences[a] = [question_id]
        paragraph_questions.append(question_id)
        question_id +=1

      answer_context_array = [context for s in sentences]
      sentences = list(sentences)
      sentence_embeddings = module.signatures['response_encoder'](input=tf.constant(sentences), context=tf.constant(answer_context_array))['outputs']
      for i in range(0,len(sentences)):
        s = sentences[i]
        if s in answer_sentences:
          feed_file.write(json.dumps(make_vespa_feed(sentence_id,answer_sentences[s],s,sentence_embeddings[i],context_id)))
          feed_file.write("\n")
        else:
          feed_file.write(json.dumps(make_vespa_feed(sentence_id,[],s,sentence_embeddings[i],context_id)))
          feed_file.write("\n")
        sentence_id +=1
    
      feed_file.write(json.dumps(make_vespa_feed_paragraph(paragraph_questions,context,context_id)))
      feed_file.write("\n")
      context_id +=1

def chunks(l, n):
  for i in range(0, len(l), n):
    yield l[i:i + n]

#Create embedding for queries
chunks = chunks(queries,200)
for chunk in chunks: 
  chunk_queries = [str(q[1]) for q in chunk]
  embeddings = module.signatures['question_encoder'](tf.constant(chunk_queries)) ['outputs']
  for i in range(0,len(chunk)):
    question_id,question,number_answers = chunk[i]
    query_file.write("%i\t%s\t%i\t%s\n" % (int(question_id),str(question),int(number_answers),str(embeddings[i].numpy().tolist())))

query_file.close()
feed_file.close()
