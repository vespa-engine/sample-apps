import os
import json
import random
import codecs


def decode_rot13(encoded_text):
    return codecs.decode(encoded_text, "rot_13")


def create_zen_json_files(n, output_dir="data"):
    """
    Creates n JSON files (named 1.json, 2.json, ..., n.json) in the given output directory.
    Each file has the format:
    {
        "put": "id:doc:doc::{i}",
        "fields": {
            "text": "...randomly sampled 100 words..."
        }
    }

    :param n: Number of JSON files to create
    :param output_dir: Directory where the files will be saved (default: .data/)
    """
    # Import the Zen of Python text
    import this

    folder_to_field = {"inside": "text", "sidecar": "text_sidecar"}

    # Get all words from the Zen of Python
    zen_text = this.s
    plain_text = decode_rot13(zen_text)
    words = plain_text.split()

    # Create n files
    for i in range(1, n + 1):
        # Randomly pick 100 words (with replacement, so words can repeat)
        sampled_words = random.choices(words, k=100)
        text = " ".join(sampled_words)
        for folder_ind, (folder, field) in enumerate(folder_to_field.items()):
            # Create doc id
            doc_id = i + (n * folder_ind)
            # Ensure the output directory exists
            os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
            # Construct the data structure
            data = {"put": f"id:doc:doc::{doc_id}", "fields": {field: text}}

            # Write to JSON file
            file_path = os.path.join(output_dir, folder, f"{doc_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)


# Example usage:
if __name__ == "__main__":
    create_zen_json_files(1000, output_dir="data/")
