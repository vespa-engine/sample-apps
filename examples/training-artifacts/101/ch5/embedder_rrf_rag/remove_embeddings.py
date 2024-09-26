import json

INPUT_FILE="ext/vespa_feed-1K.jsonl"
OUTPUT_FILE="ext/vespa_feed-1K_no_embeddings.jsonl"

# example line in the input file
# {"put": "id:product:product::Amazon_Fashion987231", "fields": {"id": 987231, "title": "CENTER Rabbit Party Hats Christmas Hats for Adults Bunny Hat Rabbit Ears Mobile Jumping Cap Fun Hat", "category": "Amazon_Fashion", "description": "", "price": 3100, "average_rating": 4.5, "embedding": [0.008706144988536835, 0.00692428182810545, ...]}}

# remove the "embedding" field from each line
with open(INPUT_FILE, "r") as input_file:
    with open(OUTPUT_FILE, "w") as output_file:
        for line in input_file:
            line_dict = json.loads(line)
            del line_dict["fields"]["embedding"]
            output_file.write(json.dumps(line_dict) + "\n")
