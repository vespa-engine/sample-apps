import pandas as pd
from dotenv import load_dotenv
import os
import base64
from PIL import Image
import io
from datasets import Dataset, Image as HFImage
from pathlib import Path
from tqdm import tqdm

load_dotenv()

df = pd.read_json("output/vespa_feed_full.jsonl", lines=True)
df = pd.json_normalize(df["fields"].tolist())

dataset_dir = Path("hf_dataset")
image_dir = dataset_dir / "images"
os.makedirs(image_dir, exist_ok=True)


def save_image(image_data, filename):
    img_data = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_data))
    img.save(filename)


for idx, row in tqdm(df.iterrows()):
    blur_filename = os.path.join(image_dir, f"blur_{idx}.jpg")
    full_filename = os.path.join(image_dir, f"full_{idx}.jpg")
    save_image(row["blur_image"], blur_filename)
    save_image(row["full_image"], full_filename)
    df.at[idx, "blur_image"] = blur_filename
    df.at[idx, "full_image"] = full_filename


# Step 3: Convert to Hugging Face Dataset
dataset = (
    Dataset.from_dict(df.to_dict(orient="list"))
    .cast_column("blur_image", HFImage())
    .cast_column("full_image", HFImage())
)
dataset.push_to_hub("vespa-engine/gpfg-QA", private=True)
