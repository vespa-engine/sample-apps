"""
Compare GTE-ModernColBERT ONNX inference outputs with Vespa's computed similarities.

This script:
1. Loads the ONNX model and tokenizer
2. Embeds the query and documents using the same tokenization as Vespa
3. Computes the element-wise similarity (all_sims)
4. Compares with Vespa's output from resp.json

Key differences to account for:
- Vespa uses bfloat16 for ColBERT embeddings, ONNX uses float32
- Vespa pads query tokens to 32 with [MASK] tokens (ColBERT-style query augmentation)
- Tokenization may differ slightly for punctuation
"""

import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# ==========================================
# Configuration
# ==========================================

MODEL_REPO = "thomasht86/GTE-ModernColBERT-v1-onnx"
QUERY = "exchanging information by sound"
VESPA_RESPONSE_FILE = "resp.json"

# Vespa ColBERT query expansion settings
QUERY_TOKEN_LENGTH = 32  # Vespa pads queries to 32 tokens with [MASK]

# Special token IDs used by Vespa's ColBERT embedder (from services.xml)
CLS_TOKEN_ID = 50281  # [CLS]
SEP_TOKEN_ID = 50282  # [SEP]
MASK_TOKEN_ID = 50284  # [MASK]
PAD_TOKEN_ID = 50283  # [PAD]
QUERY_TOKEN_ID = 50368  # [Q]
DOCUMENT_TOKEN_ID = 50369  # [D]

# Punctuation characters that Vespa filters from document tokens
# See ColBertEmbedder.java: PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

# ==========================================
# Load Model and Tokenizer
# ==========================================

print(f"Loading ONNX model and tokenizer from {MODEL_REPO}...")

# Download ONNX model
onnx_path = hf_hub_download(MODEL_REPO, "gte_moderncolbert.onnx")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

# Create ONNX session
session = ort.InferenceSession(onnx_path)

print("Model loaded successfully.")

# ==========================================
# Load Vespa Response
# ==========================================

print(f"\nLoading Vespa response from {VESPA_RESPONSE_FILE}...")

with open(VESPA_RESPONSE_FILE, "r") as f:
    vespa_response = json.load(f)

# Extract documents from response
documents = []
for child in vespa_response["root"]["children"]:
    doc_id = child["id"]
    text = child["fields"]["text"]
    all_sims = child["fields"]["matchfeatures"]["all_sims"]
    max_sim = child["fields"]["matchfeatures"]["max_sim"]
    documents.append(
        {
            "id": doc_id,
            "text": text,
            "all_sims": all_sims,
            "max_sim_vespa": max_sim,
        }
    )

print(f"Found {len(documents)} documents:")
for doc in documents:
    print(f'  - {doc["id"]}: "{doc["text"]}"')

# ==========================================
# Tokenization Functions (Vespa-style)
# ==========================================


def tokenize_query(text: str, tokenizer, pad_to: int = QUERY_TOKEN_LENGTH) -> tuple:
    """
    Tokenize query with Vespa's ColBERT style:
    [CLS] [Q] token1 token2 ... [SEP] [MASK] [MASK] ... (padded to pad_to tokens)

    ColBERT pads queries with [MASK] tokens to enable query augmentation.
    Note: Attention mask is 1 for real tokens and 0 for [MASK] padding tokens.
    See Vespa ColBertEmbedder.java buildTransformerInput() method.
    """
    # Tokenize text (without special tokens)
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Build input: [CLS] [Q] tokens... [SEP]
    input_ids = [CLS_TOKEN_ID, QUERY_TOKEN_ID] + tokens + [SEP_TOKEN_ID]
    input_length = len(input_ids)

    # Pad with [MASK] tokens to reach pad_to length (ColBERT query augmentation)
    num_masks = pad_to - input_length
    if num_masks > 0:
        input_ids = input_ids + [MASK_TOKEN_ID] * num_masks

    # Attention mask: 1 for real tokens, 0 for [MASK] padding tokens
    # This matches Vespa's behavior where padding tokens are not attended to
    attention_mask = [1] * input_length + [0] * max(0, num_masks)

    return np.array([input_ids], dtype=np.int64), np.array(
        [attention_mask], dtype=np.int64
    )


def get_punctuation_token_ids(tokenizer) -> set:
    """
    Get the set of token IDs that correspond to punctuation characters.
    Vespa filters these from document embeddings.
    """
    skip_tokens = set()
    for char in PUNCTUATION:
        token_ids = tokenizer.encode(char, add_special_tokens=False)
        skip_tokens.update(token_ids)
    return skip_tokens


def tokenize_document(text: str, tokenizer, skip_tokens: set = None) -> tuple:
    """
    Tokenize document with Vespa's ColBERT style:
    [CLS] [D] token1 token2 ... [SEP]

    Note: Vespa filters out punctuation tokens from document embeddings.
    """
    # Tokenize text (without special tokens)
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Filter out punctuation tokens (Vespa does this for documents)
    if skip_tokens is not None:
        tokens = [t for t in tokens if t not in skip_tokens]

    # Build input: [CLS] [D] tokens... [SEP]
    input_ids = [CLS_TOKEN_ID, DOCUMENT_TOKEN_ID] + tokens + [SEP_TOKEN_ID]
    attention_mask = [1] * len(input_ids)

    return np.array([input_ids], dtype=np.int64), np.array(
        [attention_mask], dtype=np.int64
    )


def float32_to_bfloat16_simulation(arr: np.ndarray) -> np.ndarray:
    """
    Simulate bfloat16 precision by truncating float32 mantissa.
    bfloat16 has same exponent as float32 but only 7 bits of mantissa (vs 23).
    """
    # Convert to uint32 view, mask off lower 16 bits of mantissa, convert back
    arr_copy = arr.copy()  # Don't modify original
    arr_uint32 = arr_copy.view(np.uint32)
    arr_uint32 &= 0xFFFF0000  # Keep sign, exponent, and upper 7 bits of mantissa
    return arr_uint32.view(np.float32)


def get_embeddings(
    input_ids: np.ndarray, attention_mask: np.ndarray, simulate_bfloat16: bool = False
) -> np.ndarray:
    """
    Run ONNX inference to get token embeddings.

    Args:
        simulate_bfloat16: If True, truncate to bfloat16 precision to match Vespa storage.
                          Note: Vespa stores doc embeddings as bfloat16, query as float.
    """
    outputs = session.run(
        ["token_embeddings"], {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    embeddings = outputs[0]  # Shape: (1, seq_len, 128)

    if simulate_bfloat16:
        embeddings = float32_to_bfloat16_simulation(embeddings)

    return embeddings


# ==========================================
# Initialize Punctuation Filter
# ==========================================

# Get punctuation token IDs that Vespa filters from documents
skip_tokens = get_punctuation_token_ids(tokenizer)
print(f"Punctuation token IDs to skip: {skip_tokens}")

# ==========================================
# Compute Query Embeddings
# ==========================================

print(f'\n--- Processing Query: "{QUERY}" ---')

query_input_ids, query_attention_mask = tokenize_query(QUERY, tokenizer)
# Query embeddings - Vespa also stores these as bfloat16 in the query tensor
query_embeddings = get_embeddings(
    query_input_ids, query_attention_mask, simulate_bfloat16=True
)

print(f"Query tokens: {query_input_ids.shape[1]}")
print(f"Query embeddings shape: {query_embeddings.shape}")

# Decode query tokens for reference
query_token_ids = query_input_ids[0].tolist()
query_tokens = []
for tid in query_token_ids:
    if tid == CLS_TOKEN_ID:
        query_tokens.append("[CLS]")
    elif tid == QUERY_TOKEN_ID:
        query_tokens.append("[Q]")
    elif tid == SEP_TOKEN_ID:
        query_tokens.append("[SEP]")
    elif tid == MASK_TOKEN_ID:
        query_tokens.append("[MASK]")
    else:
        query_tokens.append(tokenizer.decode([tid]))
print(f"Query tokens: {query_tokens}")

# ==========================================
# Compare with Each Document
# ==========================================

print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)

for doc in documents:
    print(f"\n--- Document: {doc['id']} ---")
    print(f'Text: "{doc["text"]}"')

    # Tokenize and embed document
    # Vespa stores document embeddings as bfloat16, so simulate that precision
    # Vespa also filters out punctuation tokens from documents
    doc_input_ids, doc_attention_mask = tokenize_document(
        doc["text"], tokenizer, skip_tokens=skip_tokens
    )
    doc_embeddings = get_embeddings(
        doc_input_ids, doc_attention_mask, simulate_bfloat16=True
    )

    print(f"Document tokens: {doc_input_ids.shape[1]}")

    # Decode document tokens for reference
    doc_token_ids = doc_input_ids[0].tolist()
    doc_tokens = []
    for tid in doc_token_ids:
        if tid == CLS_TOKEN_ID:
            doc_tokens.append("[CLS]")
        elif tid == DOCUMENT_TOKEN_ID:
            doc_tokens.append("[D]")
        elif tid == SEP_TOKEN_ID:
            doc_tokens.append("[SEP]")
        else:
            doc_tokens.append(tokenizer.decode([tid]))
    print(f"Document tokens: {doc_tokens}")

    # Compute all_sims: element-wise product of query and doc embeddings
    # Shape: query_embeddings (1, qt, 128), doc_embeddings (1, dt, 128)
    # Result: all_sims (qt, dt, 128)
    q_emb = query_embeddings[0]  # (qt, 128)
    d_emb = doc_embeddings[0]  # (dt, 128)

    # Compute element-wise product for all (qt, dt) pairs
    # all_sims[qt, dt, x] = q_emb[qt, x] * d_emb[dt, x]
    local_all_sims = q_emb[:, np.newaxis, :] * d_emb[np.newaxis, :, :]  # (qt, dt, 128)

    print(f"Computed all_sims shape: {local_all_sims.shape}")

    # Parse Vespa's all_sims
    vespa_all_sims = doc["all_sims"]
    vespa_blocks = vespa_all_sims["blocks"]

    # Reconstruct Vespa's tensor from blocks
    num_qt = len(set(b["address"]["qt"] for b in vespa_blocks))
    num_dt = len(set(b["address"]["dt"] for b in vespa_blocks))

    print(f"Vespa all_sims dimensions: qt={num_qt}, dt={num_dt}")

    # Create array to hold Vespa values
    vespa_sims = np.zeros((num_qt, num_dt, 128), dtype=np.float32)
    for block in vespa_blocks:
        qt_idx = int(block["address"]["qt"])
        dt_idx = int(block["address"]["dt"])
        vespa_sims[qt_idx, dt_idx, :] = np.array(block["values"], dtype=np.float32)

    # Compare shapes
    if local_all_sims.shape != vespa_sims.shape:
        print(
            f"⚠️  Shape mismatch! Local: {local_all_sims.shape}, Vespa: {vespa_sims.shape}"
        )
        # Trim to smaller shape for comparison
        min_qt = min(local_all_sims.shape[0], vespa_sims.shape[0])
        min_dt = min(local_all_sims.shape[1], vespa_sims.shape[1])
        local_trimmed = local_all_sims[:min_qt, :min_dt, :]
        vespa_trimmed = vespa_sims[:min_qt, :min_dt, :]
    else:
        local_trimmed = local_all_sims
        vespa_trimmed = vespa_sims

    # Compute difference statistics
    diff = np.abs(local_trimmed - vespa_trimmed)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print("\nElement-wise comparison (all_sims):")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")

    # Compute max_sim locally
    # max_sim = max over qt of (max over dt of (sum over x of all_sims))
    token_sims = np.sum(local_all_sims, axis=2)  # (qt, dt)
    max_per_qt = np.max(token_sims, axis=1)  # (qt,)
    local_max_sim = np.max(max_per_qt)

    print("\nmax_sim comparison:")
    print(f"  Local max_sim:  {local_max_sim:.10f}")
    print(f"  Vespa max_sim:  {doc['max_sim_vespa']:.10f}")
    print(f"  Difference:     {abs(local_max_sim - doc['max_sim_vespa']):.6e}")

    # Show sample values for verification
    print("\nSample all_sims values (qt=0, dt=0, first 5 dims):")
    print(f"  Local:  {local_all_sims[0, 0, :5]}")
    print(f"  Vespa:  {vespa_sims[0, 0, :5]}")

    if max_diff < 0.01:
        print("\n✅ Values match within tolerance!")
    else:
        print("\n❌ Significant differences detected")

# ==========================================
# Summary
# ==========================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f'Query: "{QUERY}"')
print(f"Query tokens (with MASK padding): {query_input_ids.shape[1]}")
print(f"Documents compared: {len(documents)}")
print("\nThis comparison verifies that:")
print("  1. The ONNX model produces the same embeddings as Vespa")
print("  2. The tokenization (special tokens) matches Vespa's ColBERT embedder")
print("  3. Query padding with [MASK] tokens matches Vespa's ColBERT configuration")
print("  4. bfloat16 precision simulation applied to match Vespa's storage format")
print("  5. The element-wise similarity computation matches")
print("\nNote: Small differences may occur due to:")
print("  - bfloat16 vs float32 precision")
print("  - Potential tokenization differences for special characters/punctuation")
