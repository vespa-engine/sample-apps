import sys
import re
import json
import os

file_name = 'src/main/resources/files/accepted_words.txt'

def read_json_file(file_path):
    data = None
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def read_common_words(file_path):
    common_words = set()
    with open(file_path, 'r') as f:
       lines = f.readlines()
       for line in lines:
           common_words.add(line.strip())
    return common_words

def clean_text(text):
    return re.split(r'[^a-z0-9]+', text.lower())

def process_docs(obj):
    term_set = set()
    for doc in obj:
        if doc['fields']['title'] and doc['fields']['content']:
            for term in clean_text(doc['fields']['title']):
                term_set.add(term)
            for term in clean_text(doc['fields']['content']):
                term_set.add(term)
    return term_set

def remove_stop_words(term_set, stop_word_set):
    term_set.difference_update(stop_word_set)

def write_to_file(term_set):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        for term in term_set:
            f.write(term+"\n")

def main():
    common_words = set()
    data = read_json_file(sys.argv[1])
    if len(sys.argv) == 3:
        common_words = read_common_words(sys.argv[2])
    term_set = process_docs(data)
    remove_stop_words(term_set, common_words)
    write_to_file(term_set)


if __name__ == '__main__':
    main()

