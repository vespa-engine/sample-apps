#!/usr/bin/env python3

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from io import BytesIO
from typing import cast

from colpali_engine.models import ColPali, ColPaliProcessor

from colpali_engine.utils.torch_utils import get_torch_device
from vidore_benchmark.utils.image_utils import scale_image, get_base64_image
import requests
from pdf2image import convert_from_path
from pypdf import PdfReader
import numpy as np
from vespa.application import Vespa


def main():
    parser = argparse.ArgumentParser(description="Feed data into Vespa application")
    parser.add_argument("--vespa_app_url", required=True, help="Vespa application URL")
    parser.add_argument(
        "--vespa_cloud_secret_token", required=True, help="Vespa Cloud secret token"
    )
    args = parser.parse_args()

    vespa_app_url = args.vespa_app_url
    vespa_cloud_secret_token = args.vespa_cloud_secret_token

    model_name = "vidore/colpali-v1.2"
    processor_name = "google/paligemma-3b-mix-448"

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
    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(processor_name))

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
            collate_fn=lambda x: process_images(processor, x),
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
            embedding_dict = dict()
            for idx, patch_embedding in enumerate(embedding):
                binary_vector = (
                    np.packbits(np.where(patch_embedding > 0, 1, 0))
                    .astype(np.int8)
                    .tobytes()
                    .hex()
                )
                embedding_dict[idx] = binary_vector
            page = {
                "id": str(hash(url + str(page_number))),
                "url": url,
                "title": title,
                "page_number": page_number,
                "image": base_64_image,
                "text": page_text,
                "embedding": embedding_dict,
            }
            vespa_feed.append(page)

    # Instantiate Vespa connection using token
    app = Vespa(url=vespa_app_url, vespa_cloud_secret_token=vespa_cloud_secret_token)

    def callback(response):
        if response.status_code != 200:
            print(response.text)

    # Feed data into Vespa

    app.feed_async_iterable(vespa_feed, callback=callback)


if __name__ == "__main__":
    main()
