#!/usr/bin/env python3
# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import pandas
import sys
import json

def get(df_row, key, defaultValue):
  value = df_row[key] 
  if value == 'notvalid':
    return defaultValue
  else:
    return value
  

def to_tally(row):
  return {
      'doi':  'https://doi.org/%s' % get(row, 'doi', None),
      'total': get(row, 'total', None),
      'supporting': get(row, 'supporting', None),
      'contradicting': get(row, 'contradicting', None),
      'mentioning': get(row, 'mentioning', None),
      'unclassified': get(row, 'unclassified', None)
  }

def to_citation(row):
  return {
      'source_doi': 'https://doi.org/%s' % get(row, 'source_doi', None),
      'target_doi': 'https://doi.org/%s' % get(row, 'target_doi', None),
      'context': get(row, 'text', None),
      'pos': get(row, 'pos', None),
      'neg': get(row, 'neg', None),
      'neu': get(row, 'neu', None),
      'type': get(row, 'type', None)
  }

DATA_FILE = sys.argv[1]
TALLIES_FILE = sys.argv[2]
CITATIONS_FILE = sys.argv[3]

docs = {}
with open(DATA_FILE, 'r') as f:
  for doc in json.load(f):
    doc['citations_supporting'] = []
    doc['citations_contradicting'] = []
    doc['citations_inbound'] = []
    doc['citations_outbound'] = []
    doc['citations_count_total'] = 0
    doc['citations_count_supporting'] = 0
    doc['citations_count_contradicting'] = 0
    doc['citations_sum_positive'] = 0
    doc['citations_sum_negative'] = 0
    doc['citations_sum_neutral'] = 0
    docs[doc['doi']] = doc

tallies = pandas.read_csv(TALLIES_FILE)
tallies = tallies.fillna("notvalid")
tallies_assigned = 0
tallies_not_assigned = 0
for _, row in tallies.iterrows():
  tally = to_tally(row)
  if tally['doi'] in docs.keys():
    doc = docs[tally['doi']]
    doc['citations_count_total'] = tally['total']
    doc['citations_count_supporting'] = tally['supporting']
    doc['citations_count_contradicting'] = tally['contradicting']
    tallies_assigned = tallies_assigned + 1
  else:
    #print("No document found for tally with doi %s" % tally['doi'], file=sys.stderr)
    tallies_not_assigned = tallies_not_assigned + 1
print("Tallies assigned: %d, not assigned :%d" % (tallies_assigned, tallies_not_assigned), file=sys.stderr)

citations = pandas.read_csv(CITATIONS_FILE)
citations = citations.fillna("notvalid")
citation_targets_assigned = 0
citation_targets_not_assigned = 0
citation_sources_assigned = 0
citation_sources_not_assigned = 0
for _, row in citations.iterrows():
  citation = to_citation(row)
  #print("Processing citation %s --> %s" % (citation['source_doi'], citation['target_doi']), file=sys.stderr)
  if citation['target_doi'] in docs.keys():
    doc = docs[citation['target_doi']]
    citation['target_id'] = doc['id']
    doc['citations_inbound'].append(citation)
    doc['citations_sum_positive'] = doc['citations_sum_positive'] + citation['pos']
    doc['citations_sum_negative'] = doc['citations_sum_negative'] + citation['neg']
    doc['citations_sum_neutral'] = doc['citations_sum_neutral'] + citation['neu']
    if citation['type'] == 'supporting':
      doc['citations_supporting'].append(citation['context'])
    if citation['type'] == 'contradicting':
      doc['citations_contradicting'].append(citation['context'])
    citation_targets_assigned = citation_targets_assigned + 1
  else:
    #print("No document found for citation target doi %s" % citation['target_doi'], file=sys.stderr)
    citation_targets_not_assigned = citation_targets_not_assigned + 1
  if citation['source_doi'] in docs.keys():
    doc = docs[citation['source_doi']]
    doc['citations_outbound'].append(citation)
    citation['source_id'] = doc['id']
    citation_sources_assigned = citation_sources_assigned + 1
  else:
    #print("No document found for citation source doi %s" % citation['source_doi'], file=sys.stderr)
    citation_sources_not_assigned = citation_sources_not_assigned + 1
print("Source citations assigned: %d, not assigned :%d" % (citation_sources_assigned, citation_sources_not_assigned), file=sys.stderr)
print("Target citations assigned: %d, not assigned :%d" % (citation_targets_assigned, citation_targets_not_assigned), file=sys.stderr)

print(json.dumps(list(docs.values())))
