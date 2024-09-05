import os
from requests import get
import streamlit as st
from math import floor
from vespa.application import Vespa
from embedding import VideoSearchApp, TextProcessor

st.set_page_config(layout="wide")

VESPA_URL = os.environ.get("VESPA_ENDPOINT", "http://localhost:8080")
VESPA_CERT_PATH = os.environ.get("VESPA_CERT_PATH", None)
try:
    VIDEO_FOLDER = os.environ["VIDEO_DIR"]
except KeyError:
    raise ValueError("Set VIDEO_DIR env variable")

app = Vespa(
    url=VESPA_URL,
    cert=VESPA_CERT_PATH,
)


@st.cache
def get_text_processor():
    text_processor = TextProcessor(model_name="ViT-B/32")
    return text_processor


def get_video(video_file_name, video_dir):
    return os.path.join(video_dir, video_file_name)


def get_predefined_queries():
    return get(
        "https://data.vespa-cloud.com/blog/ucf101/predefined_queries.txt"
    ).text.split("\n")[:-1]


out1, col1, out2 = st.columns([3, 1, 3])
col1.image("https://docs.vespa.ai/assets/vespa-logo-color.png", width=100)
flag_predefined_queries = st.checkbox(label="Use pre-defined queries?")
if flag_predefined_queries:
    query_input = st.selectbox(
        label="Predefined queries", options=get_predefined_queries()
    )
else:
    query_input = st.text_input(
        label="Query", value="a baby crawling", key="query_input"
    )


video_app = VideoSearchApp(app=app, text_processor=get_text_processor())
results = video_app.query(text=query_input, number_videos=6)
video_file_names = [x["video_file_name"] for x in results]

number_rows = floor(len(video_file_names) / 3)
remainder = len(video_file_names) % 3
if number_rows > 0:
    for i in range(number_rows):
        col1, col2, col3 = st.columns(3)
        col1.video(get_video(video_file_names[3 * i], video_dir=VIDEO_FOLDER))
        col2.video(get_video(video_file_names[3 * i + 1], video_dir=VIDEO_FOLDER))
        col3.video(get_video(video_file_names[3 * i + 2], video_dir=VIDEO_FOLDER))
if remainder > 0:
    cols = st.columns(3)
    for i in range(remainder):
        cols[i].video(
            get_video(video_file_names[3 * number_rows + i], video_dir=VIDEO_FOLDER)
        )
