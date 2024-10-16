# %% [markdown]
# # Prepare and Feed Vespa Application
#

# %% [markdown]
# ## 0. Setup and Configuration
#

# %%
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
    AuthClient,
    Parameter,
    FieldSet,
    SecondPhaseRanking,
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


# %% [markdown]
# ## Create a free trial in Vespa Cloud ($300 credit)
#
# Create a tenant from [here](https://vespa.ai/free-trial/).
# Take note of your tenant name.
#

# %%
VESPA_TENANT_NAME = "vespa-team"

# %% [markdown]
# Here, set your desired application name. (Will be created in later steps)
# Note that you can not have hyphen `-` or underscore `_` in the application name.
#

# %%
VESPA_APPLICATION_NAME = "colpalidemo"

# %% [markdown]
# Next, you need to create some tokens for feeding data, and querying the application.
# We recommend separate tokens for feeding and querying, (the former with write permission, and the latter with read permission).
# The tokens can be created from the [Vespa Cloud console](https://console.vespa-cloud.com/) in the 'Account' -> 'Tokens' section.
#

# %%
VESPA_TOKEN_ID_WRITE = "colpalidemo_write"
VESPA_TOKEN_ID_READ = "colpalidemo_read"

# %% [markdown]
# We also need to set the value of the token.
#

# %%
VESPA_CLOUD_SECRET_TOKEN = os.getenv("VESPA_CLOUD_SECRET_TOKEN") or input(
    "Enter Vespa cloud secret token: "
)

# %% [markdown]
# We will also use the Gemini API to create sample queries for our images.
# You can also use other VLM's to create these queries.
# Create a Gemini API key from [here](https://aistudio.google.com/app/apikey).
#

# %%
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or input(
    "Enter Google Generative AI API key: "
)

# %%
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

# %% [markdown]
# ## 1. Download PDFs
#
# We are
#

# %%
import requests

url = "https://www.nbim.no/en/publications/reports/"
response = requests.get(url)
response.raise_for_status()
html_content = response.text

# Parse with BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")

links = []

# Find all <a> elements with the specific classes
for a_tag in soup.find_all("a", href=True):
    classes = a_tag.get("class", [])
    if "button" in classes and "button--download-secondary" in classes:
        href = a_tag["href"]
        full_url = urljoin(url, href)
        links.append(full_url)

links

# %%
from nest_asyncio import apply
from typing import List

apply()


async def download_pdf(session, url, filename):
    attempt = 0
    while True:
        attempt += 1
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
            return
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            print(f"Retrying ({attempt})...")
            await asyncio.sleep(1)  # Wait a bit before retrying


async def download_pdfs(links: List[str]):
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

        await asyncio.gather(*tasks)


# Create the pdfs directory if it doesn't exist
os.makedirs("pdfs", exist_ok=True)
# Now run the download_pdfs function with the URL
asyncio.run(download_pdfs(links))

# %% [markdown]
# ## 2. Convert PDFs to Images
#

# %%


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
pdf_files = [
    os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")
]
# TODO: Do full
pdf_files = pdf_files[:5]
pdf_pages = []
for pdf_file in tqdm(pdf_files):
    title = os.path.splitext(os.path.basename(pdf_file))[0]
    url = pdf_file  # In this case, the local file path
    images, texts = get_pdf_images(pdf_file)
    for page_no, (image, text) in enumerate(zip(images, texts)):
        pdf_pages.append(
            {
                "title": title,
                "url": url,
                "image": image,
                "text": text,
                "page_no": page_no,
            }
        )

# %%
len(pdf_pages)

# %%
type(images[0])

# %% [markdown]
# ## 3. Generate Queries (Using Google Generative AI - Gemini1.5b)
#

# %%
from pydantic import BaseModel


class GeneralRetrievalQuery(BaseModel):
    broad_topical_query: str
    broad_topical_explanation: str
    specific_detail_query: str
    specific_detail_explanation: str
    visual_element_query: str
    visual_element_explanation: str


def get_retrieval_prompt() -> Tuple[str, GeneralRetrievalQuery]:
    prompt = """You are an AI assistant specialized in document retrieval tasks. Given an image of a document page, your task is to generate retrieval queries that someone might use to find this document in a large corpus.
Please generate 3 different types of retrieval queries:
1. A broad topical query: This should cover the main subject of the document.
2. A specific detail query: This should focus on a particular fact, figure, or point made in the document.
3. A visual element query: This should reference a chart, graph, image, or other visual component in the document, if present. Don't just reference the name of the visual element but generate a query which this illustration may help answer or be related to.
Important guidelines:
- Ensure the queries are relevant for retrieval tasks, not just describing the page content.
- Frame the queries as if someone is searching for this document, not asking questions about its content.
- Make the queries diverse and representative of different search strategies.
For each query, also provide a brief explanation of why this query would be effective in retrieving this document.
Format your response as a JSON object with the following structure:
{
  "broad_topical_query": "Your query here",
  "broad_topical_explanation": "Brief explanation",
  "specific_detail_query": "Your query here",
  "specific_detail_explanation": "Brief explanation",
  "visual_element_query": "Your query here",
  "visual_element_explanation": "Brief explanation"
}
If there are no relevant visual elements, replace the third query with another specific detail query.
Here is the document image to analyze:
Generate the queries based on this image and provide the response in the specified JSON format.
Only return JSON. Don't return any extra explanation text."""
    return prompt, GeneralRetrievalQuery


prompt_text, pydantic_model = get_retrieval_prompt()

# %%
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai

max_in_flight = 5  # Maximum number of concurrent requests


async def generate_queries_for_image_async(model, image, semaphore):
    @retry(stop=stop_after_attempt(3), wait=wait_exponential())
    async def _generate():
        async with semaphore:
            result = await model.generate_content_async(
                [image, "\n\n", prompt_text],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=pydantic_model,
                ),
            )
            return json.loads(result.text)

    try:
        return await _generate()
    except Exception as e:
        print(f"Error generating queries for image: {e}")
        return None  # Return None or handle as needed


model = genai.GenerativeModel("gemini-1.5-pro-latest")
semaphore = asyncio.Semaphore(max_in_flight)
tasks = []
for pdf in pdf_pages:
    pdf["queries"] = []
    image = pdf.get("image")
    if image:
        task = asyncio.create_task(
            generate_queries_for_image_async(model, image, semaphore)
        )
        tasks.append((pdf, task))

# Run the tasks
for pdf, task in tqdm(tasks):
    queries = await task
    if queries:
        pdf["queries"].append(queries)

# %%
len(pdf_pages)

# %%
# Unique titles in pdf_pages

unique_titles = set([pdf["title"] for pdf in pdf_pages])
len(unique_titles)

# %%
unique_titles

# %%
title = "gpfg_annual-report-2022"

# Filter pdf_pages by title
pdf_pages_filtered = filter(lambda x: x["title"] == title, pdf_pages)
pdf_pages_filtered = list(pdf_pages_filtered)

# %%
pdf_pages_filtered

# %% [markdown]
# ## 4. Prepare Data in Vespa Format
#

# %%
# We will prepare the Vespa feed data, including the embeddings and the generated queries


def generate_embeddings(images):
    page_embeddings = []
    dataloader = DataLoader(
        images,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    for batch_doc in tqdm(dataloader, desc="Generating embeddings"):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
            page_embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    return page_embeddings


vespa_feed = []
for pdf in sample_pdfs:
    url = pdf["url"]
    title = pdf["title"]
    images = pdf["images"]
    texts = pdf["texts"]
    queries_list = pdf.get("queries", [])
    embeddings = generate_embeddings(images)
    for page_number, (page_text, embedding, image, queries) in enumerate(
        zip(texts, embeddings, images, queries_list)
    ):
        base_64_image = get_base64_image(scale_image(image, 640), add_url_prefix=False)
        base_64_full_image = get_base64_image(image, add_url_prefix=False)
        embedding_dict = dict()
        for idx, patch_embedding in enumerate(embedding):
            binary_vector = (
                np.packbits(np.where(patch_embedding > 0, 1, 0))
                .astype(np.int8)
                .tobytes()
                .hex()
            )
            embedding_dict[idx] = binary_vector
        # id_hash should be md5 hash of url and page_number
        id_hash = hashlib.md5(f"{url}_{page_number}".encode()).hexdigest()
        page = {
            "id": id_hash,
            "fields": {
                "id": id_hash,
                "url": url,
                "title": title,
                "page_number": page_number,
                "image": base_64_image,
                "full_image": base_64_full_image,
                "text": page_text,
                "embedding": embedding_dict,
                "queries": queries,
            },
        }
        vespa_feed.append(page)

# Save vespa_feed to vespa_feed.json
with open("vespa_feed.json", "w") as f:
    vespa_feed_to_save = []
    for page in vespa_feed:
        document_id = page["id"]
        put_id = f"id:{VESPA_APPLICATION_NAME}:pdf_page::{document_id}"
        vespa_feed_to_save.append({"put": put_id, "fields": page["fields"]})
    json.dump(vespa_feed_to_save, f)

# %% [markdown]
# ## 5. Prepare Vespa Application
#

# %%
# Define the Vespa schema
colpali_schema = Schema(
    name="pdf_page",
    document=Document(
        fields=[
            Field(
                name="id",
                type="string",
                indexing=["summary", "index"],
                match=["word"],
            ),
            Field(name="url", type="string", indexing=["summary", "index"]),
            Field(
                name="title",
                type="string",
                indexing=["summary", "index"],
                match=["text"],
                index="enable-bm25",
            ),
            Field(name="page_number", type="int", indexing=["summary", "attribute"]),
            Field(name="image", type="raw", indexing=["summary"]),
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
                ],  # adds HNSW index for candidate retrieval.
                ann=HNSW(
                    distance_metric="hamming",
                    max_links_per_node=32,
                    neighbors_to_explore_at_insert=400,
                ),
            ),
            Field(
                name="queries",
                type="json",
                indexing=["summary"],
            ),
        ]
    ),
    fieldsets=[
        FieldSet(name="default", fields=["title", "url", "page_number", "text"]),
        FieldSet(name="image", fields=["image"]),
    ],
)

# Define rank profiles
colpali_profile = RankProfile(
    name="default",
    inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
    functions=[
        Function(
            name="max_sim",
            expression="""
                sum(
                    reduce(
                        sum(
                            query(qt) * unpack_bits(attribute(embedding)) , v
                        ),
                        max, patch
                    ),
                    querytoken
                )
            """,
        ),
        Function(name="bm25_score", expression="bm25(title) + bm25(text)"),
    ],
    first_phase="bm25_score",
    second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=10),
)
colpali_schema.add_rank_profile(colpali_profile)

# Add retrieval-and-rerank rank profile
input_query_tensors = []
MAX_QUERY_TERMS = 64
for i in range(MAX_QUERY_TERMS):
    input_query_tensors.append((f"query(rq{i})", "tensor<int8>(v[16])"))

input_query_tensors.append(("query(qt)", "tensor<float>(querytoken{}, v[128])"))
input_query_tensors.append(("query(qtb)", "tensor<int8>(querytoken{}, v[16])"))

colpali_retrieval_profile = RankProfile(
    name="retrieval-and-rerank",
    inputs=input_query_tensors,
    functions=[
        Function(
            name="max_sim",
            expression="""
                sum(
                    reduce(
                        sum(
                            query(qt) * unpack_bits(attribute(embedding)) , v
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
                    1/(1 + sum(
                        hamming(query(qtb), attribute(embedding)) ,v)
                    ),
                    max,
                    patch
                  ),
                  querytoken
                )
            """,
        ),
    ],
    first_phase="max_sim_binary",
    second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=10),
)
colpali_schema.add_rank_profile(colpali_retrieval_profile)

# Create the Vespa application package
vespa_application_package = ApplicationPackage(
    name=VESPA_APPLICATION_NAME,
    schema=[colpali_schema],
    auth_clients=[
        AuthClient(
            id="mtls",  # Note that you still need to include the mtls client.
            permissions=["read", "write"],
            parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
        ),
        AuthClient(
            id="token_write",
            permissions=["read", "write"],
            parameters=[Parameter("token", {"id": VESPA_TOKEN_ID_WRITE})],
        ),
        AuthClient(
            id="token_read",
            permissions=["read"],
            parameters=[Parameter("token", {"id": VESPA_TOKEN_ID_READ})],
        ),
    ],
)

# Save the application package to disk
application_package_dir = f"{VESPA_APPLICATION_NAME}-package"
vespa_application_package.to_files(dir_path=application_package_dir)

# %% [markdown]
# ## 6. Deploy Vespa Application
#

# %%
# Collect constants and environment variables


VESPA_TEAM_API_KEY = os.getenv("VESPA_TEAM_API_KEY") or input(
    "Enter Vespa team API key: "
)
VESPA_APP_URL = os.getenv("VESPA_APP_URL") or input("Enter Vespa application URL: ")


# %%
vespa_cloud = VespaCloud(
    tenant=VESPA_TENANT_NAME,
    application=VESPA_APPLICATION_NAME,
    key_content=VESPA_TEAM_API_KEY,
    application_package=vespa_application_package,
    output_dir=application_package_dir,
)

# Deploy the application
vespa_cloud.deploy()

# Output the endpoint URL
endpoint_url = vespa_cloud.get_token_endpoint()
print(f"Application deployed. Token endpoint URL: {endpoint_url}")

# %% [markdown]
# ## 7. Prepare Data on Vespa Format
#

# %%
# Data has already been prepared in the 'vespa_feed' variable and saved to 'vespa_feed.json'

# %% [markdown]
# ## 8. Feed Data to Vespa
#

# %%
# Instantiate Vespa connection using token
app = Vespa(url=VESPA_APP_URL, vespa_cloud_secret_token=VESPA_CLOUD_SECRET_TOKEN)
app.get_application_status()

# Load vespa_feed from vespa_feed.json
with open("vespa_feed.json", "r") as f:
    vespa_feed_saved = json.load(f)
vespa_feed = []
for doc in vespa_feed_saved:
    put_id = doc["put"]
    fields = doc["fields"]
    # Extract document_id from put_id
    # Format: 'id:application_name:schema_name::document_id'
    parts = put_id.split("::")
    document_id = parts[1] if len(parts) > 1 else ""
    page = {"id": document_id, "fields": fields}
    vespa_feed.append(page)


def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(
            f"Failed to feed document {id} with status code {response.status_code}: Reason {response.get_json()}"
        )


# Feed data into Vespa
app.feed_iterable(vespa_feed, schema="pdf_page", callback=callback)
