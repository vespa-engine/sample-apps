#!/usr/bin/env python3
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import sys
import json
import spacy
import re
import json
import mmh3
nlp = spacy.load("en_core_web_sm")
suggestions = dict()

filters = {
  ";": 1,
  "*": 1,
	"&gt":1,
	"gt;":1,
	"&lt":1,
	"lt;":1,
	"{":1,
	"}":1,
	"(":1,
	")":1,
	"<":1,
	">":1,
	":":1,
	"=":1,
	"\"":1,
	"\'":1,
	"%":1,
	"$":1,
	"191":1,
	"24":1,
	"]":1,
	"[":1,
	"[":1,
	"|":1,
	"e.g":1,
	"1.":1,
	"2.":1,
	"3.":1,
	"4.":1,
	"5.":1
}

invalid_starts = {
	"a ":1,
	"an ":1,
	"any ":1,
	"another ":1,
	"the ":1,
	"either ": 1,
	"more ": 1,
	"only ": 1
}

def filter(text):
	if len(text) < 3 or len(text) > 64:
		return True
	if text.startswith("/") and len(text) < 3:
		return True
	for f in filters:
		if f in text:
			return True
	return False

def filter_content(text):
	for f in invalid_starts:
		if text.startswith(f):
			return True
	return False

def clean_text(text):
	text = text.strip()
	text = text.lower()
	text = text.replace("\"","")
	return " ".join(text.split())

with open(sys.argv[1]) as fp:
	docs = json.load(fp)
	for doc in docs:
		fields = doc['fields']
		title = clean_text(fields['title'])
		title = " ".join(re.split(r"[^a-z0-9]+",title)[0:4])
		if not filter(title):
			suggestions[title] = 1

vocab = dict()	
for k,v in suggestions.items():	
	chunks = re.split(r"[^a-z0-9]+",k)
	for c in chunks:
		if c in vocab:
			vocab[c] = vocab[c] +1 
		else:
			vocab[c] = 1

with open(sys.argv[1]) as fp:
	docs = json.load(fp)
	for doc in docs:
		fields = doc['fields']
		content = fields['title']
		doc = nlp(content)
		for chunk in doc.noun_chunks:
			noun_phrase = clean_text(chunk.text) 
			if filter(noun_phrase):
				continue
			words = len(noun_phrase.split())
			if words < 3 or words > 5:
				continue
			if filter_content(noun_phrase):
				continue
			for v in vocab.keys():
				if v in noun_phrase:
					if noun_phrase in suggestions:
						suggestions[noun_phrase] = suggestions[noun_phrase] + 1	
					else:
						suggestions[noun_phrase] = 1 
					break


suggest = []
for k,v in suggestions.items():
	id = mmh3.hash(k)
	words = re.split(r"[^a-z0-9]+",k)
	doc = {
		'put': 'id:query:query::%i' % id,
    	'fields': {
      	'query': k,
      	'score': v,
      	'words': words
      }
  } 
	print(json.dumps(doc))	


	
