#!/usr/bin/env python3

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from io import BytesIO
from typing import cast
import os
import json
import hashlib

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from vidore_benchmark.utils.image_utils import scale_image, get_base64_image
import requests
from pdf2image import convert_from_path
from pypdf import PdfReader
import numpy as np
from vespa.application import Vespa
from vespa.io import VespaResponse
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Feed data into Vespa application")
    parser.add_argument(
        "--application_name",
        required=True,
        default="colpalidemo",
        help="Vespa application name",
    )
    parser.add_argument(
        "--vespa_schema_name",
        required=True,
        default="pdf_page",
        help="Vespa schema name",
    )
    args = parser.parse_args()

    vespa_app_url = os.getenv("VESPA_APP_URL")
    vespa_cloud_secret_token = os.getenv("VESPA_CLOUD_SECRET_TOKEN")
    # Set application and schema names
    application_name = args.application_name
    schema_name = args.vespa_schema_name
    # Instantiate Vespa connection using token
    app = Vespa(url=vespa_app_url, vespa_cloud_secret_token=vespa_cloud_secret_token)
    app.get_application_status()
    model_name = "vidore/colpali-v1.2"

    device = get_torch_device("auto")
    print(f"Using device: {device}")

    # Load the model
    model = cast(
        ColPali,
        ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        ),
    ).eval()

    # Load the processor
    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))

    # Define functions to work with PDFs
    def download_pdf(url):
        response = requests.get(url)
        if response.status_code == 200:
            return BytesIO(response.content)
        else:
            raise Exception(
                f"Failed to download PDF: Status code {response.status_code}"
            )

    def get_pdf_images(pdf_url):
        # Download the PDF
        pdf_file = download_pdf(pdf_url)
        # Save the PDF temporarily to disk (pdf2image requires a file path)
        temp_file = "temp.pdf"
        with open(temp_file, "wb") as f:
            f.write(pdf_file.read())
        reader = PdfReader(temp_file)
        page_texts = []
        for page_number in range(len(reader.pages)):
            page = reader.pages[page_number]
            text = page.extract_text()
            page_texts.append(text)
        images = convert_from_path(temp_file)
        assert len(images) == len(page_texts)
        return (images, page_texts)

    # Define sample PDFs
    sample_pdfs = [
        {
            "title": "ConocoPhillips Sustainability Highlights - Nature (24-0976)",
            "url": "https://static.conocophillips.com/files/resources/24-0976-sustainability-highlights_nature.pdf",
        },
        {
            "title": "ConocoPhillips Managing Climate Related Risks",
            "url": "https://static.conocophillips.com/files/resources/conocophillips-2023-managing-climate-related-risks.pdf",
        },
        {
            "title": "ConocoPhillips 2023 Sustainability Report",
            "url": "https://static.conocophillips.com/files/resources/conocophillips-2023-sustainability-report.pdf",
        },
    ]

    # Check if vespa_feed.json exists
    if os.path.exists("vespa_feed.json"):
        print("Loading vespa_feed from vespa_feed.json")
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
    else:
        print("Generating vespa_feed")
        # Process PDFs
        for pdf in sample_pdfs:
            page_images, page_texts = get_pdf_images(pdf["url"])
            pdf["images"] = page_images
            pdf["texts"] = page_texts

        # Generate embeddings
        for pdf in sample_pdfs:
            page_embeddings = []
            dataloader = DataLoader(
                pdf["images"],
                batch_size=2,
                shuffle=False,
                collate_fn=lambda x: processor.process_images(x),
            )
            for batch_doc in tqdm(dataloader):
                with torch.no_grad():
                    batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
                    embeddings_doc = model(**batch_doc)
                    page_embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
            pdf["embeddings"] = page_embeddings

        # Prepare Vespa feed
        vespa_feed = []
        for pdf in sample_pdfs:
            url = pdf["url"]
            title = pdf["title"]
            for page_number, (page_text, embedding, image) in enumerate(
                zip(pdf["texts"], pdf["embeddings"], pdf["images"])
            ):
                base_64_image = get_base64_image(
                    scale_image(image, 640), add_url_prefix=False
                )
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
                    },
                }
                vespa_feed.append(page)

        # Save vespa_feed to vespa_feed.json in the specified format
        vespa_feed_to_save = []
        for page in vespa_feed:
            document_id = page["id"]
            put_id = f"id:{application_name}:{schema_name}::{document_id}"
            vespa_feed_to_save.append({"put": put_id, "fields": page["fields"]})
        with open("vespa_feed.json", "w") as f:
            json.dump(vespa_feed_to_save, f)

    def callback(response: VespaResponse, id: str):
        if not response.is_successful():
            print(
                f"Failed to feed document {id} with status code {response.status_code}: Reason {response.get_json()}"
            )

    # Feed data into Vespa
    app.feed_iterable(vespa_feed, schema=schema_name, callback=callback)


if __name__ == "__main__":
    main()
