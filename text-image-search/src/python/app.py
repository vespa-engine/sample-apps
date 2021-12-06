import os
import clip
import time
import streamlit as st
from math import floor
from PIL import Image

from vespa.application import Vespa
from vespa.query import QueryModel, ANN, QueryRankingFeature, RankProfile as Ranking
from embedding import (
    translate_model_names_to_valid_vespa_field_names,
    TextProcessor,
    decode_string_to_media,
)

st.set_page_config(layout="wide")

VESPA_URL = os.environ.get("VESPA_ENDPOINT", "http://localhost:8080")
VESPA_CERT_PATH = os.environ.get("VESPA_CERT_PATH", None)

app = Vespa(
    url=VESPA_URL,
    cert=VESPA_CERT_PATH,
)


@st.cache(ttl=24 * 60 * 60)
def get_available_clip_model_names():
    return clip.available_models()


@st.cache(ttl=7 * 24 * 60 * 60)
def get_text_processor(clip_model_name):
    text_processor = TextProcessor(model_name=clip_model_name)
    return text_processor


def get_image(image_file_name, image_dir):
    return Image.open(os.path.join(image_dir, image_file_name))


def vespa_query(query, clip_model_name):
    vespa_model_name = translate_model_names_to_valid_vespa_field_names(clip_model_name)
    image_vector_name = vespa_model_name + "_image"
    text_vector_name = vespa_model_name + "_text"
    ranking_name = vespa_model_name + "_similarity"
    text_processor = get_text_processor(clip_model_name=clip_model_name)
    result = app.query(
        query=query,
        query_model=QueryModel(
            name=vespa_model_name,
            match_phase=ANN(
                doc_vector=image_vector_name,
                query_vector=text_vector_name,
                hits=100,
                label="clip_ann",
            ),
            rank_profile=Ranking(name=ranking_name, list_features=False),
            query_properties=[
                QueryRankingFeature(name=text_vector_name, mapping=text_processor.embed)
            ],
        ),
        **{"presentation.timing": "true"}
    )
    return [hit["fields"]["encoded_image"] for hit in result.hits], result.json[
        "timing"
    ]


clip_model_name = st.sidebar.selectbox(
    "Select CLIP model", get_available_clip_model_names()
)

out1, col1, out2 = st.columns([3, 1, 3])
col1.image("https://docs.vespa.ai/assets/vespa-logo-color.png", width=100)
query_input = st.text_input(label="", value="a man surfing", key="query_input")

start = time.time()
images, timing = vespa_query(query=query_input, clip_model_name=clip_model_name)
placeholder = st.empty()
number_rows = floor(len(images) / 3)
remainder = len(images) % 3
if number_rows > 0:
    for i in range(number_rows):
        col1, col2, col3 = st.columns(3)
        col1.image(decode_string_to_media(images[3 * i]))
        col2.image(decode_string_to_media(images[3 * i + 1]))
        col3.image(decode_string_to_media(images[3 * i + 2]))
if remainder > 0:
    cols = st.columns(3)
    for i in range(remainder):
        cols[i].image(decode_string_to_media(images[3 * number_rows + i]))
total_timing = time.time() - start
vespa_search_time = round(timing["searchtime"], 2)
total_time = round(total_timing, 2)
other_time = round(total_time - vespa_search_time, 2)
placeholder.write(
    "**Vespa search time: {}s**. Network related time: {}s. Total time: {}s".format(
        vespa_search_time, other_time, total_time
    )
)

