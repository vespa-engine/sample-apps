import os
import clip
import streamlit as st
from math import floor
from PIL import Image

from vespa.application import Vespa
from vespa.query import QueryModel, ANN, QueryRankingFeature, RankProfile as Ranking
from embedding import translate_model_names_to_valid_vespa_field_names, TextProcessor

st.set_page_config(layout="wide")

VESPA_URL = os.environ.get("VESPA_ENDPOINT", "http://localhost:8080")
VESPA_CERT_PATH = os.environ.get("VESPA_CERT_PATH", None)
try:
    IMG_FOLDER = os.environ["IMG_DIR"]
except KeyError:
    raise ValueError("Set IMG_DIR env variable")

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
    )
    return [hit["fields"]["image_file_name"] for hit in result.hits]


clip_model_name = st.sidebar.selectbox(
    "Select CLIP model", get_available_clip_model_names()
)

out1, col1, out2 = st.columns([3,1,3])
col1.image("https://docs.vespa.ai/assets/vespa-logo-color.png", width=100)
query_input = st.text_input(label="", value="a man surfing", key="query_input")

image_file_names = vespa_query(query=query_input, clip_model_name=clip_model_name)

number_rows = floor(len(image_file_names)/3)
remainder = len(image_file_names) % 3
if number_rows > 0:
    for i in range(number_rows):
        col1, col2, col3 = st.columns(3)
        col1.image(get_image(image_file_names[3*i], image_dir=IMG_FOLDER))
        col2.image(get_image(image_file_names[3*i + 1], image_dir=IMG_FOLDER))
        col3.image(get_image(image_file_names[3*i + 2], image_dir=IMG_FOLDER))
if remainder > 0:
    cols = st.columns(3)
    for i in range(remainder):
        cols[i].image(get_image(image_file_names[3*number_rows+i], image_dir=IMG_FOLDER))

