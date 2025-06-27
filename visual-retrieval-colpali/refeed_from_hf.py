import base64
import os
from io import BytesIO

from datasets import load_dataset
from dotenv import load_dotenv
from vespa.application import Vespa
from vespa.io import VespaResponse

load_dotenv()

VESPA_APP_TOKEN_URL = os.getenv("VESPA_APP_TOKEN_URL")
VESPA_CLOUD_SECRET_TOKEN = os.getenv("VESPA_CLOUD_SECRET_TOKEN")

ds = load_dataset("vespa-engine/gpfg-QA", split="train", streaming=True)

app = Vespa(
    url=VESPA_APP_TOKEN_URL,
    vespa_cloud_secret_token=VESPA_CLOUD_SECRET_TOKEN,
)
app.get_application_status()


def pil_to_base64(pil_image):
    """Convert PIL image to base64 string"""
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_base64


def process_item_for_vespa(item):
    """Convert dataset item to Vespa format with base64 images and proper embedding structure"""
    # Convert PIL images to base64
    blur_image_b64 = pil_to_base64(item["blur_image"])
    full_image_b64 = pil_to_base64(item["full_image"])

    # Reconstruct embedding dictionary from separate embedding.X fields
    embedding_dict = {}
    for key, value in item.items():
        if key.startswith("embedding."):
            # Extract the number after "embedding."
            embedding_index = int(key.split(".")[1])
            embedding_dict[embedding_index] = value

    # Return Vespa-formatted item
    return {
        "id": item["id"],
        "fields": {
            "id": item["id"],
            "url": item["url"],
            "title": item["title"],
            "year": item["year"],
            "page_number": item["page_number"],
            "blur_image": blur_image_b64,
            "full_image": full_image_b64,
            "text": item["text"],
            "embedding": embedding_dict,
            "queries": item["queries"],
            "questions": item["questions"],
        },
    }


# Map the dataset to convert to Vespa format
vespa_feed = ds.map(process_item_for_vespa)


def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(
            f"Failed to feed document {id} with status code {response.status_code}: Reason {response.get_json()}"
        )


# Feed data into Vespa asynchronously
app.feed_async_iterable(vespa_feed, schema="pdf_page", callback=callback)
