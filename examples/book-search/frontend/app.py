import json
from pathlib import Path

import streamlit as st
from vespa.application import Vespa

VESPA_URL  = "http://localhost"
VESPA_PORT = 8080
SCHEMA     = "book"
NAMESPACE  = "library"

ALL_THEMES = ["adventure", "classic", "comedy", "coming-of-age", "cyberpunk",
              "detective", "dystopia", "fantasy", "mystery", "philosophy",
              "political", "psychological", "sci-fi", "social", "tragedy", "young-adult"]

app = Vespa(url=VESPA_URL, port=VESPA_PORT)

def feed_catalog():
    docs_path = Path(__file__).parent.parent / "data" / "documents.jsonl"
    with open(docs_path) as f:
        for line in f:
            doc    = json.loads(line)
            doc_id = doc["put"].split("::")[-1]
            app.feed_data_point(schema=SCHEMA, data_id=doc_id,
                                fields=doc["fields"], namespace=NAMESPACE)


def set_loaned_out(doc_id: str, loaned_out: bool):
    app.update_data(
        schema=SCHEMA,
        data_id=doc_id,
        fields={"loaned_out": loaned_out},
        namespace=NAMESPACE,
    )


# Auto-load catalog on first run
check = app.query(yql="select * from book where true", hits=0)
if check.json["root"]["fields"]["totalCount"] == 0:
    with st.spinner("Loading library catalog..."):
        feed_catalog()

st.title("Library Search")

query      = st.text_input("Search by title or author", "")
themes     = st.multiselect("Filter by theme", ALL_THEMES)
year_range = st.slider("Publication year", 1800, 2030, (1800, 2030))
only_avail = st.checkbox("Available only")

conditions = [f"year >= {year_range[0]}", f"year <= {year_range[1]}"]
for theme in themes:
    conditions.append(f'themes contains "{theme}"')
if only_avail:
    conditions.append("loaned_out = false")
if query:
    conditions.insert(0, "userInput(@query)")

where  = " and ".join(conditions)
params = {"yql": f"select * from book where {where}", "hits": 20}
if query:
    params["query"] = query

response = app.query(**params)
hits     = response.hits
total    = response.json["root"]["fields"]["totalCount"]
st.caption(f"{total} book(s) in library — showing {len(hits)}")

for hit in hits:
    f       = hit["fields"]
    doc_id  = hit["id"].split("::")[-1]
    on_loan = f.get("loaned_out", False)

    col1, col2 = st.columns([4, 1])
    with col1:
        status = "On loan" if on_loan else "Available"
        st.markdown(f"**{f['title']}** — {f['author']} ({f['year']})")
        st.caption(f"{status} · {', '.join(f.get('themes', []))}")
    with col2:
        if on_loan:
            if st.button("Return", key=doc_id):
                set_loaned_out(doc_id, False)
                st.rerun()
        else:
            if st.button("Loan Out", key=doc_id):
                set_loaned_out(doc_id, True)
                st.rerun()
