import streamlit as st

OPENAI_API_KEY = st.secrets["api_keys"]["llm"]
TAVILY_API_KEY = st.secrets["api_keys"]["tavily"]
VESPA_URL = st.secrets["vespa"]["url"]
PUBLIC_CERT_PATH = st.secrets["vespa"]["public_cert_path"]
PRIVATE_KEY_PATH = st.secrets["vespa"]["private_key_path"]