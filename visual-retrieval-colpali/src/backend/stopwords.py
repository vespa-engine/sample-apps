import spacy
import os

# Download the model if it is not already present
if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")


# It would be possible to remove bolding for stopwords without removing them from the query,
# but that would require a java plugin which we didn't want to complicate this sample app with.
def filter(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]
    if len(tokens) == 0:
        # if we remove all the words we don't have a query at all, so use the original
        return text
    return " ".join(tokens)
