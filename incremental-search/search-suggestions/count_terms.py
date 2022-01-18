import sys
import re
import json


def read_json_file(file_path):
    data = None
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def get_phrases(terms):
    # from "learning to rank" to ['learning to rank', 'to rank', 'rank']
    phrases = []
    phrases.append(terms)
    end = terms.find(' ')
    while end != -1:
        start = end+1
        remainder = terms[start:]
        phrases.append(remainder)
        end = terms.find(' ', start)
    return phrases

def write_json_file(terms, file_path):
    feed_list = [
        {
            "update": f'id:term:term::{re.sub(r" ", "/", term)}',
            "create": True,
            "fields": {
                "term": {"assign": data["term"]},
                "terms": {"assign": get_phrases(data["term"])},
                "corpus_count": {"assign": data["count"]},
                "document_count": {"assign": data["docs"]},
            },
        }
        for (term, data) in terms.items()
    ]
    with open(file_path, "w") as f:
        json.dump(feed_list, f, indent=2)


def read_common_words(file_path):
    common_words = set()
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            common_words.add(line.strip())
    return common_words


def clean_text(text):
    return re.split(r"[^a-z0-9]+", text.lower())


def count_terms(doc, term_length=1, common_words=set()):
    counts = {}
    for win_len in range(1, term_length + 1):
        for i in range(len(doc) + 1 - win_len):
            if not any(
                map(lambda word: word in common_words, doc[i : i + win_len])
            ):
                term = " ".join(doc[i : i + win_len])
                if  len(term) > 0 and not term.isspace():
                    counts[term] = counts.get(term, 0) + 1
    return counts


def process_doc(doc_fields, term_length=1, common_words=set()):
    if doc_fields["title"] and doc_fields["content"]:
        title_counts = count_terms(
            clean_text(doc_fields["title"]), term_length, common_words
        )
        content_counts = count_terms(
            clean_text(doc_fields["content"]), term_length, common_words
        )
        return {
            term: title_counts.get(term, 0) + content_counts.get(term, 0)
            for term in list(title_counts.keys()) + list(content_counts.keys())
        }
    else:
        return {}


def process_docs(obj, term_length=1, common_words=set()):
    doc_counts = [
        process_doc(doc["fields"], term_length, common_words) for doc in obj
    ]
    terms = {}

    # Sum counts
    for counts in doc_counts:
        total_count = sum([count for count in counts.values()])
        for (term, count) in counts.items():
            if term in terms:
                terms[term]["count"] = terms[term]["count"] + count
            else:
                terms[term] = {"term": term, "count": count}

    # Find number of documents containing the term
    for term in terms.keys():
        terms[term]["docs"] = sum(
            [1 if term in doc else 0 for doc in doc_counts]
        )

    return terms


def main():
    if len(sys.argv) == 3:
        write_json_file(process_docs(read_json_file(sys.argv[1])), sys.argv[2])
    elif len(sys.argv) == 4:
        write_json_file(
            process_docs(read_json_file(sys.argv[1]), int(sys.argv[3])),
            sys.argv[2],
        )
    elif len(sys.argv) == 5:
        write_json_file(
            process_docs(
                read_json_file(sys.argv[1]),
                int(sys.argv[3]),
                read_common_words(sys.argv[4]),
            ),
            sys.argv[2],
        )
    else:
        print(
            "Wrong number of arguments:",
            "python3",
            "count_terms.py",
            "infile.json",
            "outfile.json",
            "[number_of_words_in_term [common_words.txt]]",
        )


if __name__ == "__main__":
    main()
