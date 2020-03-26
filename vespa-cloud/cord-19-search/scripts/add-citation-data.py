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
      'doi': get(row, 'doi', None),
      'total': get(row, 'total', None),
      'supporting': get(row, 'supporting', None),
      'contradicting': get(row, 'contradicting', None),
      'mentioning': get(row, 'mentioning', None),
      'unclassified': get(row, 'unclassified', None)
  }

def to_citation(row):
  return {
      'source_doi': get(row, 'source_doi', None),
      'target_doi': get(row, 'target_doi', None),
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
    docs[doc['raw_doi']] = doc

tallies = pandas.read_csv(TALLIES_FILE)
tallies = tallies.fillna("notvalid")
for _, row in tallies.iterrows():
  tally = to_tally(row)
  if tally['doi'] in docs.keys():
    doc = docs[tally['doi']]
    doc['citations_count_total'] = tally['total']
    doc['citations_count_supporting'] = tally['supporting']
    doc['citations_count_contradicting'] = tally['contradicting']

citations = pandas.read_csv(CITATIONS_FILE)
citations = citations.fillna("notvalid")
for _, row in citations.iterrows():
  citation = to_citation(row)
  if citation['target_doi'] in docs.keys():
    doc = docs[citation['target_doi']]
    doc['citations_inbound'].append(citation)
    doc['citations_sum_positive'] = doc['citations_sum_positive'] + citation['pos']
    doc['citations_sum_negative'] = doc['citations_sum_negative'] + citation['neg']
    doc['citations_sum_neutral'] = doc['citations_sum_neutral'] + citation['neu']
    if citation['type'] == 'supporting':
      doc['citations_supporting'].append(citation['context'])
    if citation['type'] == 'contradicting':
      doc['citations_contradicting'].append(citation['context'])
  if citation['source_doi'] in docs.keys():
    docs[citation['source_doi']]['citations_outbound'].append(citation)

print(json.dumps(list(docs.values())))
