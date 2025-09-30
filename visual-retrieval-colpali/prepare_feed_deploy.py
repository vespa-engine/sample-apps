# # Visual PDF Retrieval - demo application
#
# In this notebook, we will prepare the Vespa backend application for our visual retrieval demo.
# We will use ColPali as the model to extract patch vectors from images of pdf pages.
# At query time, we use MaxSim to retrieve and/or (based on the configuration) rank the page results.
#
# To see the application in action, visit https://huggingface.co/spaces/vespa-engine/colpali-vespa-visual-retrieval
#
# The web application is written in FastHTML, meaning the complete application is written in python.
#
# The steps we will take in this notebook are:
#
# 0. Setup and configuration
# 1. Download the data
# 2. Prepare the data
# 3. Generate queries for evaluation and typeahead search suggestions
# 4. Deploy the Vespa application
# 5. Create the Vespa application
# 6. Feed the data to the Vespa application
#
# All the steps that are needed to provision the Vespa application, including feeding the data, can be done from this notebook.
# We have tried to make it easy for others to run this notebook, to create your own PDF Enterprise Search application using Vespa.
#

# ## 0. Setup and Configuration
#

# +
import os
import asyncio
import json
from typing import Tuple
import hashlib
import numpy as np

# Vespa
from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    HNSW,
    RankProfile,
    Function,
    FieldSet,
    SecondPhaseRanking,
    Summary,
    DocumentSummary,
)
from vespa.deployment import VespaCloud
from vespa.application import Vespa
from vespa.io import VespaResponse

# Google Generative AI
import google.generativeai as genai

# Torch and other ML libraries
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pdf2image import convert_from_path
from pypdf import PdfReader

# ColPali model and processor
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from vidore_benchmark.utils.image_utils import scale_image, get_base64_image

# Other utilities
from bs4 import BeautifulSoup
import httpx
from urllib.parse import urljoin, urlparse

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Avoid warning from huggingface tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# -

# ### Create a free trial in Vespa Cloud
#
# Create a tenant from [here](https://vespa.ai/free-trial/).
# The trial includes $300 credit.
# Take note of your tenant name.
#

VESPA_TENANT_NAME = "vespa-team"

# Here, set your desired application name. (Will be created in later steps)
# Note that you can not have hyphen `-` or underscore `_` in the application name.
#

VESPA_APPLICATION_NAME = "colpalidemo"
VESPA_SCHEMA_NAME = "pdf_page"

# Next, you need to create some tokens for feeding data, and querying the application.
# We recommend separate tokens for feeding and querying, (the former with write permission, and the latter with read permission).
# The tokens can be created from the [Vespa Cloud console](https://console.vespa-cloud.com/) in the 'Account' -> 'Tokens' section.
#

VESPA_TOKEN_ID_WRITE = "colpalidemo_write"

# We also need to set the value of the write token to be able to feed data to the Vespa application.
#

VESPA_CLOUD_SECRET_TOKEN = os.getenv("VESPA_CLOUD_SECRET_TOKEN") or input(
    "Enter Vespa cloud secret token: "
)

# We will also use the Gemini API to create sample queries for our images.
# You can also use other VLM's to create these queries.
# Create a Gemini API key from [here](https://aistudio.google.com/app/apikey).
#

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or input(
    "Enter Google Generative AI API key: "
)

# +
MODEL_NAME = "vidore/colpali-v1.2"

# Configure Google Generative AI
genai.configure(api_key=GEMINI_API_KEY)

# Set device for Torch
device = get_torch_device("auto")
print(f"Using device: {device}")

# Load the ColPali model and processor
model = ColPali.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map=device,
).eval()

processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
# -

# ## 1. Download PDFs
#
# We are going to use public reports from the Norwegian Government Pension Fund Global (also known as the Oil Fund).
# The fund puts transparency at the forefront and publishes reports on its investments, holdings, and returns, as well as its strategy and governance.
#
# These reports are the ones we are going to use for this showcase.
# Here are some sample images:
#
# ![Sample1](./static/img/gfpg-sample-1.png)
# ![Sample2](./static/img/gfpg-sample-2.png)
#

# As we can see, a lot of the information is in the form of tables, charts and numbers.
# These are not easily extractable using pdf-readers or OCR tools.
#

# +
import requests

url = "https://www.nbim.no/en/publications/reports/"
response = requests.get(url)
response.raise_for_status()
html_content = response.text

# Parse with BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")

links = []
url_to_year = {}

# Find all the years
for year_section in soup.find_all("section", attrs={"data-name": "report-year"}):
    year_id = year_section.get("data-filter-section", "")
    year = year_id.replace("year-", "")
    
    # Find the yearly report section
    report_section = year_section.find("div", attrs={
        "data-filter-section": "year",
        "data-name": "report-type"
    })
    if not report_section:
        continue
        
    # Get the first link
    report_link = report_section.select_one("ul.link-list a")
    if not report_link:
        continue
        
    report_url = urljoin(url, report_link["href"])
    
    # Visit the report page to find PDF download links
    try:
        report_response = requests.get(report_url)
        report_response.raise_for_status()
        report_soup = BeautifulSoup(report_response.text, "html.parser")
        
        # Find only the first PDF download link with the specific class
        pdf_link = report_soup.select_one("a.btn.btn-secondary[data-right-icon='download']")
        if not pdf_link or not pdf_link["href"].endswith(".pdf"):
            continue
            
        pdf_url = urljoin(report_url, pdf_link["href"])
        links.append(pdf_url)
        url_to_year[pdf_url] = year
        print(f"Found PDF: {pdf_url} (Year: {year})")
    except Exception as e:
        print(f"Error fetching report page {report_url}: {e}")

print(f"Found {len(links)} PDF links")
# -

# Limit the number of PDFs to download
NUM_PDFS = 2  # Set to None to download all PDFs
links = links[:NUM_PDFS] if NUM_PDFS else links
links

# +
from nest_asyncio import apply
from typing import List

apply()

max_attempts = 3


async def download_pdf(session, url, filename):
    attempt = 0
    while attempt < max_attempts:
        try:
            response = await session.get(url)
            response.raise_for_status()

            # Use Content-Disposition header to get the filename if available
            content_disposition = response.headers.get("Content-Disposition")
            if content_disposition:
                import re

                fname = re.findall('filename="(.+)"', content_disposition)
                if fname:
                    filename = fname[0]

            # Ensure the filename is safe to use on the filesystem
            safe_filename = filename.replace("/", "_").replace("\\", "_")
            if not safe_filename or safe_filename == "_":
                print(f"Invalid filename: {filename}")
                continue

            filepath = os.path.join("pdfs", safe_filename)
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {safe_filename}")
            return filepath
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            print(f"Retrying ({attempt})...")
            await asyncio.sleep(1)  # Wait a bit before retrying
            attempt += 1
    return None


async def download_pdfs(links: List[str]) -> List[dict]:
    """Download PDFs from a list of URLs. Add the filename to the dictionary."""
    async with httpx.AsyncClient() as client:
        tasks = []

        for idx, link in enumerate(links):
            # Try to get the filename from the URL
            path = urlparse(link).path
            filename = os.path.basename(path)

            # If filename is empty,skip
            if not filename:
                continue
            tasks.append(download_pdf(client, link, filename))

        # Run the tasks concurrently
        paths = await asyncio.gather(*tasks)
        pdf_files = [
            {"url": link, "path": path} for link, path in zip(links, paths) if path
        ]
        return pdf_files


# Create the pdfs directory if it doesn't exist
os.makedirs("pdfs", exist_ok=True)
# Now run the download_pdfs function with the URL
pdfs = asyncio.run(download_pdfs(links))
# -

pdfs

# ## 2. Convert PDFs to Images
#


# +
def get_pdf_images(pdf_path):
    reader = PdfReader(pdf_path)
    page_texts = []
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text = page.extract_text()
        page_texts.append(text)
    images = convert_from_path(pdf_path)
    # Convert to PIL images
    assert len(images) == len(page_texts)
    return images, page_texts


pdf_folder = "pdfs"
pdf_pages = []
for pdf in tqdm(pdfs):
    pdf_file = pdf["path"]
    title = os.path.splitext(os.path.basename(pdf_file))[0]
    images, texts = get_pdf_images(pdf_file)
    for page_no, (image, text) in enumerate(zip(images, texts)):
        pdf_pages.append(
            {
                "title": title,
                "year": int(url_to_year[pdf["url"]]),
                "url": pdf["url"],
                "path": pdf_file,
                "image": image,
                "text": text,
                "page_no": page_no,
            }
        )
# -

len(pdf_pages)

# +
from collections import Counter

# Print the length of the text fields - mean, max and min
text_lengths = [len(page["text"]) for page in pdf_pages]
print(f"Mean text length: {np.mean(text_lengths)}")
print(f"Max text length: {np.max(text_lengths)}")
print(f"Min text length: {np.min(text_lengths)}")
print(f"Median text length: {np.median(text_lengths)}")
print(f"Number of text with length == 0: {Counter(text_lengths)[0]}")
# -

# ## 3. Generate Queries
#
# In this step, we want to generate queries for each page image.
# These will be useful for 2 reasons:
#
# 1. We can use these queries as typeahead suggestions in the search bar.
# 2. We can use the queries to generate an evaluation dataset. See [Improving Retrieval with LLM-as-a-judge](https://blog.vespa.ai/improving-retrieval-with-llm-as-a-judge/) for a deeper dive into this topic.
#
# The prompt for generating queries is taken from [this](https://danielvanstrien.xyz/posts/post-with-code/colpali/2024-09-23-generate_colpali_dataset.html#an-update-retrieval-focused-prompt) wonderful blog post by Daniel van Strien.
#
# We will use the Gemini API to generate these queries, with `gemini-1.5-flash-8b` as the model.
#

# +
from pydantic import BaseModel


class GeneratedQueries(BaseModel):
    broad_topical_question: str
    broad_topical_query: str
    specific_detail_question: str
    specific_detail_query: str
    visual_element_question: str
    visual_element_query: str


def get_retrieval_prompt() -> Tuple[str, GeneratedQueries]:
    prompt = (
        prompt
    ) = """You are an investor, stock analyst and financial expert. You will be presented an image of a document page from a report published by the Norwegian Government Pension Fund Global (GPFG). The report may be annual or quarterly reports, or policy reports, on topics such as responsible investment, risk etc.
Your task is to generate retrieval queries and questions that you would use to retrieve this document (or ask based on this document) in a large corpus.
Please generate 3 different types of retrieval queries and questions.
A retrieval query is a keyword based query, made up of 2-5 words, that you would type into a search engine to find this document.
A question is a natural language question that you would ask, for which the document contains the answer.
The queries should be of the following types:
1. A broad topical query: This should cover the main subject of the document.
2. A specific detail query: This should cover a specific detail or aspect of the document.
3. A visual element query: This should cover a visual element of the document, such as a chart, graph, or image.

Important guidelines:
- Ensure the queries are relevant for retrieval tasks, not just describing the page content.
- Use a fact-based natural language style for the questions.
- Frame the queries as if someone is searching for this document in a large corpus.
- Make the queries diverse and representative of different search strategies.

Format your response as a JSON object with the structure of the following example:
{
    "broad_topical_question": "What was the Responsible Investment Policy in 2019?",
    "broad_topical_query": "responsible investment policy 2019",
    "specific_detail_question": "What is the percentage of investments in renewable energy?",
    "specific_detail_query": "renewable energy investments percentage",
    "visual_element_question": "What is the trend of total holding value over time?",
    "visual_element_query": "total holding value trend"
}

If there are no relevant visual elements, provide an empty string for the visual element question and query.
Here is the document image to analyze:
Generate the queries based on this image and provide the response in the specified JSON format.
Only return JSON. Don't return any extra explanation text. """

    return prompt, GeneratedQueries


prompt_text, pydantic_model = get_retrieval_prompt()

# +
gemini_model = genai.GenerativeModel("gemini-1.5-flash-8b")


def generate_queries(image, prompt_text, pydantic_model):
    try:
        response = gemini_model.generate_content(
            [image, "\n\n", prompt_text],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=pydantic_model,
            ),
        )
        queries = json.loads(response.text)
    except Exception as _e:
        queries = {
            "broad_topical_question": "",
            "broad_topical_query": "",
            "specific_detail_question": "",
            "specific_detail_query": "",
            "visual_element_question": "",
            "visual_element_query": "",
        }
    return queries


# -

for pdf in tqdm(pdf_pages):
    image = pdf.get("image")
    pdf["queries"] = generate_queries(image, prompt_text, pydantic_model)

pdf_pages[46]["image"]

pdf_pages[46]["queries"]

# +
# Generate queries async - keeping for now as we probably need when applying to the full dataset
# import asyncio
# from tenacity import retry, stop_after_attempt, wait_exponential
# import google.generativeai as genai
# from tqdm.asyncio import tqdm_asyncio

# max_in_flight = 200  # Maximum number of concurrent requests


# async def generate_queries_for_image_async(model, image, semaphore):
#     @retry(stop=stop_after_attempt(3), wait=wait_exponential(), reraise=True)
#     async def _generate():
#         async with semaphore:
#             result = await model.generate_content_async(
#                 [image, "\n\n", prompt_text],
#                 generation_config=genai.GenerationConfig(
#                     response_mime_type="application/json",
#                     response_schema=pydantic_model,
#                 ),
#             )
#             return json.loads(result.text)

#     try:
#         return await _generate()
#     except Exception as e:
#         print(f"Error generating queries for image: {e}")
#         return None  # Return None or handle as needed


# async def enrich_pdfs():
#     gemini_model = genai.GenerativeModel("gemini-1.5-flash-8b")
#     semaphore = asyncio.Semaphore(max_in_flight)
#     tasks = []
#     for pdf in pdf_pages:
#         pdf["queries"] = []
#         image = pdf.get("image")
#         if image:
#             task = generate_queries_for_image_async(gemini_model, image, semaphore)
#             tasks.append((pdf, task))

#     # Run the tasks concurrently using asyncio.gather()
#     for pdf, task in tqdm_asyncio(tasks):
#         result = await task
#         if result:
#             pdf["queries"] = result
#     return pdf_pages


# pdf_pages = asyncio.run(enrich_pdfs())

# +
# write title, url, page_no, text, queries, not image to JSON
os.makedirs("output", exist_ok=True)
with open("output/pdf_pages.json", "w") as f:
    to_write = [{k: v for k, v in pdf.items() if k != "image"} for pdf in pdf_pages]
    json.dump(to_write, f, indent=2)

# with open("pdfs/pdf_pages.json", "r") as f:
#     saved_pdf_pages = json.load(f)
# for pdf, saved_pdf in zip(pdf_pages, saved_pdf_pages):
#     pdf.update(saved_pdf)
# -

# ## 4. Generate embeddings
#
# Now that we have the queries, we can use the ColPali model to generate embeddings for each page image.
#


def generate_embeddings(images, model, processor, batch_size=2) -> np.ndarray:
    """
    Generate embeddings for a list of images.
    Move to CPU only once per batch.

    Args:
        images (List[PIL.Image]): List of PIL images.
        model (nn.Module): The model to generate embeddings.
        processor: The processor to preprocess images.
        batch_size (int, optional): Batch size for processing. Defaults to 64.

    Returns:
        np.ndarray: Embeddings for the images, shape
                    (len(images), processor.max_patch_length (1030 for ColPali), model.config.hidden_size (Patch embedding dimension - 128 for ColPali)).
    """
    embeddings_list = []

    def collate_fn(batch):
        # Batch is a list of images
        return processor.process_images(batch)  # Should return a dict of tensors

    dataloader = DataLoader(
        images,
        shuffle=False,
        collate_fn=collate_fn,
    )

    for batch_doc in tqdm(dataloader, desc="Generating embeddings"):
        with torch.no_grad():
            # Move batch to the device
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_batch = model(**batch_doc)
            embeddings_list.append(torch.unbind(embeddings_batch.to("cpu"), dim=0))
    # Concatenate all embeddings and create a numpy array
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    return all_embeddings


# Generate embeddings for all images
images = [pdf["image"] for pdf in pdf_pages]
embeddings = generate_embeddings(images, model, processor)

embeddings.shape

# ## 5. Prepare Data on Vespa Format
#
# Now, that we have all the data we need, all that remains is to make sure it is in the right format for Vespa.
#


def float_to_binary_embedding(float_query_embedding: dict) -> dict:
    """Utility function to convert float query embeddings to binary query embeddings."""
    binary_query_embeddings = {}
    for k, v in float_query_embedding.items():
        binary_vector = (
            np.packbits(np.where(np.array(v) > 0, 1, 0)).astype(np.int8).tolist()
        )
        binary_query_embeddings[k] = binary_vector
    return binary_query_embeddings


vespa_feed = []
for pdf, embedding in zip(pdf_pages, embeddings):
    url = pdf["url"]
    year = pdf["year"]
    title = pdf["title"]
    image = pdf["image"]
    text = pdf.get("text", "")
    page_no = pdf["page_no"]
    query_dict = pdf["queries"]
    questions = [v for k, v in query_dict.items() if "question" in k and v]
    queries = [v for k, v in query_dict.items() if "query" in k and v]
    base_64_image = get_base64_image(
        scale_image(image, 32), add_url_prefix=False
    )  # Scaled down image to return fast on search (~1kb)
    base_64_full_image = get_base64_image(image, add_url_prefix=False)
    embedding_dict = {k: v for k, v in enumerate(embedding)}
    binary_embedding = float_to_binary_embedding(embedding_dict)
    # id_hash should be md5 hash of url and page_number
    id_hash = hashlib.md5(f"{url}_{page_no}".encode()).hexdigest()
    page = {
        "id": id_hash,
        "fields": {
            "id": id_hash,
            "url": url,
            "title": title,
            "year": year,
            "page_number": page_no,
            "blur_image": base_64_image,
            "full_image": base_64_full_image,
            "text": text,
            "embedding": binary_embedding,
            "queries": queries,
            "questions": questions,
        },
    }
    vespa_feed.append(page)

# +
# We will prepare the Vespa feed data, including the embeddings and the generated queries


# Save vespa_feed to vespa_feed.json
with open("output/vespa_feed.json", "w") as f:
    vespa_feed_to_save = []
    for page in vespa_feed:
        document_id = page["id"]
        put_id = f"id:{VESPA_APPLICATION_NAME}:{VESPA_SCHEMA_NAME}::{document_id}"
        vespa_feed_to_save.append({"put": put_id, "fields": page["fields"]})
    json.dump(vespa_feed_to_save, f)

# +
# import json

# with open("output/vespa_feed.json", "r") as f:
#     vespa_feed = json.load(f)
# -

len(vespa_feed)

# ## 5. Prepare Vespa Application
#

# +
# Define the Vespa schema
colpali_schema = Schema(
    name=VESPA_SCHEMA_NAME,
    document=Document(
        fields=[
            Field(
                name="id",
                type="string",
                indexing=["summary", "index"],
                match=["word"],
            ),
            Field(name="url", type="string", indexing=["summary", "index"]),
            Field(name="year", type="int", indexing=["summary", "attribute"]),
            Field(
                name="title",
                type="string",
                indexing=["summary", "index"],
                match=["text"],
                index="enable-bm25",
            ),
            Field(name="page_number", type="int", indexing=["summary", "attribute"]),
            Field(name="blur_image", type="raw", indexing=["summary"]),
            Field(name="full_image", type="raw", indexing=["summary"]),
            Field(
                name="text",
                type="string",
                indexing=["summary", "index"],
                match=["text"],
                index="enable-bm25",
            ),
            Field(
                name="embedding",
                type="tensor<int8>(patch{}, v[16])",
                indexing=[
                    "attribute",
                    "index",
                ],
                ann=HNSW(
                    distance_metric="hamming",
                    max_links_per_node=32,
                    neighbors_to_explore_at_insert=400,
                ),
            ),
            Field(
                name="questions",
                type="array<string>",
                indexing=["summary", "attribute"],
                summary=Summary(fields=["matched-elements-only"]),
            ),
            Field(
                name="queries",
                type="array<string>",
                indexing=["summary", "attribute"],
                summary=Summary(fields=["matched-elements-only"]),
            ),
        ]
    ),
    fieldsets=[
        FieldSet(
            name="default",
            fields=["title", "text"],
        ),
    ],
    document_summaries=[
        DocumentSummary(
            name="default",
            summary_fields=[
                Summary(
                    name="text",
                    fields=[("bolding", "on")],
                ),
                Summary(
                    name="snippet",
                    fields=[("source", "text"), "dynamic"],
                ),
            ],
            from_disk=True,
        ),
        DocumentSummary(
            name="suggestions",
            summary_fields=[
                Summary(name="questions"),
            ],
            from_disk=True,
        ),
    ],
)

# Define similarity functions used in all rank profiles
mapfunctions = [
    Function(
        name="similarities",  # computes similarity scores between each query token and image patch
        expression="""
                sum(
                    query(qt) * unpack_bits(attribute(embedding)), v
                )
            """,
    ),
    Function(
        name="normalized",  # normalizes the similarity scores to [-1, 1]
        expression="""
                (similarities - reduce(similarities, min)) / (reduce((similarities - reduce(similarities, min)), max)) * 2 - 1
            """,
    ),
    Function(
        name="quantized",  # quantizes the normalized similarity scores to signed 8-bit integers [-128, 127]
        expression="""
                cell_cast(normalized * 127.999, int8)
            """,
    ),
]

# Define the 'bm25' rank profile
bm25 = RankProfile(
    name="bm25",
    inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
    first_phase="bm25(title) + bm25(text)",
    functions=mapfunctions,
)


# A function to create an inherited rank profile which also returns quantized similarity scores
def with_quantized_similarity(rank_profile: RankProfile) -> RankProfile:
    return RankProfile(
        name=f"{rank_profile.name}_sim",
        first_phase=rank_profile.first_phase,
        inherits=rank_profile.name,
        summary_features=["quantized"],
    )


colpali_schema.add_rank_profile(bm25)
colpali_schema.add_rank_profile(with_quantized_similarity(bm25))


# Update the 'colpali' rank profile
input_query_tensors = []
MAX_QUERY_TERMS = 64
for i in range(MAX_QUERY_TERMS):
    input_query_tensors.append((f"query(rq{i})", "tensor<int8>(v[16])"))

input_query_tensors.extend(
    [
        ("query(qt)", "tensor<float>(querytoken{}, v[128])"),
        ("query(qtb)", "tensor<int8>(querytoken{}, v[16])"),
    ]
)

colpali = RankProfile(
    name="colpali",
    inputs=input_query_tensors,
    first_phase="max_sim_binary",
    second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=10),
    functions=mapfunctions
    + [
        Function(
            name="max_sim",
            expression="""
                sum(
                    reduce(
                        sum(
                            query(qt) * unpack_bits(attribute(embedding)), v
                        ),
                        max, patch
                    ),
                    querytoken
                )
            """,
        ),
        Function(
            name="max_sim_binary",
            expression="""
                sum(
                    reduce(
                        1 / (1 + sum(
                            hamming(query(qtb), attribute(embedding)), v)
                        ),
                        max, patch
                    ),
                    querytoken
                )
            """,
        ),
    ],
)
colpali_schema.add_rank_profile(colpali)
colpali_schema.add_rank_profile(with_quantized_similarity(colpali))

# Update the 'hybrid' rank profile
hybrid = RankProfile(
    name="hybrid",
    inputs=input_query_tensors,
    first_phase="max_sim_binary",
    second_phase=SecondPhaseRanking(
        expression="max_sim + 2 * (bm25(text) + bm25(title))", rerank_count=10
    ),
    functions=mapfunctions
    + [
        Function(
            name="max_sim",
            expression="""
                sum(
                    reduce(
                        sum(
                            query(qt) * unpack_bits(attribute(embedding)), v
                        ),
                        max, patch
                    ),
                    querytoken
                )
            """,
        ),
        Function(
            name="max_sim_binary",
            expression="""
                sum(
                    reduce(
                        1 / (1 + sum(
                            hamming(query(qtb), attribute(embedding)), v)
                        ),
                        max, patch
                    ),
                    querytoken
                )
            """,
        ),
    ],
)
colpali_schema.add_rank_profile(hybrid)
colpali_schema.add_rank_profile(with_quantized_similarity(hybrid))

# +
from vespa.configuration.services import (
    services,
    container,
    search,
    document_api,
    document_processing,
    clients,
    client,
    config,
    content,
    redundancy,
    documents,
    node,
    certificate,
    token,
    document,
    nodes,
)
from vespa.configuration.vt import vt
from vespa.package import ServicesConfiguration

service_config = ServicesConfiguration(
    application_name=VESPA_APPLICATION_NAME,
    services_config=services(
        container(
            search(),
            document_api(),
            document_processing(),
            clients(
                client(
                    certificate(file="security/clients.pem"),
                    id="mtls",
                    permissions="read,write",
                ),
                client(
                    token(id=f"{VESPA_TOKEN_ID_WRITE}"),
                    id="token_write",
                    permissions="read,write",
                ),
            ),
            config(
                vt("tag")(
                    vt("bold")(
                        vt("open", "<strong>"),
                        vt("close", "</strong>"),
                    ),
                    vt("separator", "..."),
                ),
                name="container.qr-searchers",
            ),
            id=f"{VESPA_APPLICATION_NAME}_container",
            version="1.0",
        ),
        content(
            redundancy("1"),
            documents(document(type="pdf_page", mode="index")),
            nodes(node(distribution_key="0", hostalias="node1")),
            config(
                vt("max_matches", "2", replace_underscores=False),
                vt("length", "1000"),
                vt("surround_max", "500", replace_underscores=False),
                vt("min_length", "300", replace_underscores=False),
                name="vespa.config.search.summary.juniperrc",
            ),
            id=f"{VESPA_APPLICATION_NAME}_content",
            version="1.0",
        ),
        version="1.0",
    ),
)
# -

# Create the Vespa application package
vespa_application_package = ApplicationPackage(
    name=VESPA_APPLICATION_NAME,
    schema=[colpali_schema],
    services_config=service_config,
)

# ## 6. Deploy Vespa Application
#

VESPA_TEAM_API_KEY = os.getenv("VESPA_TEAM_API_KEY") or input(
    "Enter Vespa team API key: "
)

# +
vespa_cloud = VespaCloud(
    tenant=VESPA_TENANT_NAME,
    application=VESPA_APPLICATION_NAME,
    key_content=VESPA_TEAM_API_KEY,
    application_package=vespa_application_package,
)

# Deploy the application
vespa_cloud.deploy()

# Output the endpoint URL
endpoint_url = vespa_cloud.get_token_endpoint()
print(f"Application deployed. Token endpoint URL: {endpoint_url}")
# -

# Make sure to take note of the token endpoint_url.
# You need to put this in your `.env` file - `VESPA_APP_URL=https://abcd.vespa-app.cloud` - to access the Vespa application from your web application.
#

# ## 8. Feed Data to Vespa
#

# Instantiate Vespa connection using token
app = Vespa(url=endpoint_url, vespa_cloud_secret_token=VESPA_CLOUD_SECRET_TOKEN)
app.get_application_status()


# +
def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(
            f"Failed to feed document {id} with status code {response.status_code}: Reason {response.get_json()}"
        )


# Feed data into Vespa asynchronously
app.feed_async_iterable(vespa_feed, schema=VESPA_SCHEMA_NAME, callback=callback)
