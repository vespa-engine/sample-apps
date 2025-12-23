import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download, HfApi, snapshot_download
from safetensors.torch import load_file
import json
import numpy as np
import onnxruntime as ort
import shutil
import tempfile

# ==========================================
# 1. Model Definition & Loading
# ==========================================

MODEL_ID = "lightonai/GTE-ModernColBERT-v1"
ONNX_PATH = "gte_moderncolbert.onnx"
TARGET_REPO_ID = "thomasht86/GTE-ModernColBERT-v1-onnx"  # Change to your username/org

print(f"Downloading and loading components for {MODEL_ID}...")

# Load the base ModernBERT transformer
base_model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Load the projection layer (ColBERT/SentenceTransformer 'Dense' module)
print("Fetching projection layer weights...")
dense_config_path = hf_hub_download(MODEL_ID, "1_Dense/config.json")
dense_weights_path = hf_hub_download(MODEL_ID, "1_Dense/model.safetensors")

with open(dense_config_path, "r") as f:
    dense_config = json.load(f)

dense_weights = load_file(dense_weights_path)


class GTEModernColBERT(nn.Module):
    """
    GTE-ModernColBERT model for multi-vector document representations.

    This model produces per-token embeddings suitable for ColBERT-style retrieval.
    """

    def __init__(self, base_model, dense_config, dense_weights):
        super().__init__()
        self.base_model = base_model

        # Define the Linear layer based on config
        self.dense = nn.Linear(
            dense_config["in_features"],
            dense_config["out_features"],
            bias=dense_config["bias"],
        )

        # Load the weights into the linear layer
        with torch.no_grad():
            if "linear.weight" in dense_weights:
                self.dense.weight.copy_(dense_weights["linear.weight"])
                if dense_config["bias"]:
                    self.dense.bias.copy_(dense_weights["linear.bias"])
            elif "weight" in dense_weights:
                self.dense.weight.copy_(dense_weights["weight"])
                if dense_config["bias"]:
                    self.dense.bias.copy_(dense_weights["bias"])
            else:
                raise ValueError(
                    f"Could not find weight keys. Available: {dense_weights.keys()}"
                )

    def forward(self, input_ids, attention_mask):
        """Forward pass - ONNX exportable."""
        # 1. Base Transformer Forward Pass
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (Batch, Seq_Len, 768)

        # 2. Linear Projection to 128 dims
        proj = self.dense(last_hidden_state)  # (Batch, Seq_Len, 128)

        # 3. L2 Normalization (Critical for ColBERT/Cosine Similarity)
        embeddings = F.normalize(proj, p=2, dim=2)

        return embeddings


# Initialize the model
model = GTEModernColBERT(base_model, dense_config, dense_weights)
model.eval()

# ==========================================
# 2. Export to ONNX
# ==========================================

print(f"Exporting model to {ONNX_PATH}...")

# Create dummy input for export tracing
dummy_text = "Exporting ColBERT models is fun."
dummy_inputs = tokenizer(dummy_text, return_tensors="pt")
dummy_input_ids = dummy_inputs["input_ids"]
dummy_attention_mask = dummy_inputs["attention_mask"]

# Export
torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    ONNX_PATH,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input_ids", "attention_mask"],
    output_names=["token_embeddings"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "token_embeddings": {0: "batch_size", 1: "sequence_length"},
    },
    dynamo=False,  # Use legacy exporter to get single file output
)

# Convert to single file if external data was created
import os

data_file = ONNX_PATH + ".data"
if os.path.exists(data_file):
    import onnx

    print("Converting to single ONNX file (embedding external data)...")
    model_onnx = onnx.load(ONNX_PATH, load_external_data=True)
    onnx.save_model(
        model_onnx,
        ONNX_PATH,
        save_as_external_data=False,
    )
    os.remove(data_file)

print("Export complete.")

# ==========================================
# 3. Inference & Correctness Verification
# ==========================================

print("\n--- Verifying ONNX Model ---")

# Create a new inference session
ort_session = ort.InferenceSession(ONNX_PATH)

# Test Inputs
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt")
input_ids_np = inputs["input_ids"].numpy()
attention_mask_np = inputs["attention_mask"].numpy()

# 1. Run PyTorch Inference
with torch.no_grad():
    torch_output = model(inputs["input_ids"], inputs["attention_mask"])
    torch_output_np = torch_output.numpy()

# 2. Run ONNX Inference
onnx_inputs = {"input_ids": input_ids_np, "attention_mask": attention_mask_np}
onnx_output = ort_session.run(["token_embeddings"], onnx_inputs)[0]

# 3. Compare
print(f"Input Text: '{text}'")
print(f"Tokenized Shape: {input_ids_np.shape}")
print(f"Output Shape (PyTorch): {torch_output_np.shape}")
print(f"Output Shape (ONNX):   {onnx_output.shape}")

# Calculate error
diff = np.abs(torch_output_np - onnx_output)
max_diff = np.max(diff)
print(f"\nMaximum absolute difference: {max_diff:.2e}")

if max_diff < 1e-4:
    print("✅ Conversion Successful! Outputs match.")
else:
    print("❌ Conversion Failed! Outputs diverge.")

# ==========================================
# 4. Usage Example
# ==========================================
print("\n--- Usage Example ---")
tokens = tokenizer.convert_ids_to_tokens(input_ids_np[0])
fox_idx = tokens.index("fox") if "fox" in tokens else -1

if fox_idx != -1:
    fox_vector = onnx_output[0, fox_idx, :]
    print(f"Token 'fox' vector (first 5 dims): {fox_vector[:5]}")
    norm = np.linalg.norm(fox_vector)
    print(f"Vector Norm: {norm:.4f} (Should be close to 1.0)")

# ==========================================
# 5. Upload to Hugging Face Hub
# ==========================================
print("\n--- Uploading to Hugging Face Hub ---")

api = HfApi()

# Create the repository if it doesn't exist
try:
    api.create_repo(repo_id=TARGET_REPO_ID, exist_ok=True)
    print(f"Repository {TARGET_REPO_ID} ready.")
except Exception as e:
    print(f"Note: {e}")

# Create a temporary directory to stage all files
with tempfile.TemporaryDirectory() as tmpdir:
    print("Staging files in temporary directory...")

    # Download the original model files (README, tokenizer, configs, etc.)
    print(f"Downloading original model files from {MODEL_ID}...")
    original_model_dir = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=tmpdir,
        ignore_patterns=[
            "*.safetensors",  # Skip the large model weights
            "*.bin",  # Skip PyTorch bin files
            "*.pt",  # Skip PyTorch checkpoint files
            "1_Dense/*",  # Skip dense layer (already in ONNX)
        ],
    )

    # Copy the ONNX file to the staging directory
    onnx_dest = os.path.join(tmpdir, ONNX_PATH)
    shutil.copy2(ONNX_PATH, onnx_dest)
    print(f"Added {ONNX_PATH} to upload.")

    # Update/create a model card note about ONNX
    readme_path = os.path.join(tmpdir, "README.md")
    onnx_note = f"""
---

## ONNX Export

This repository contains an ONNX export of [{MODEL_ID}](https://huggingface.co/{MODEL_ID}).

### ONNX Model Details
- **File**: `{ONNX_PATH}`
- **Opset Version**: 14
- **Inputs**: `input_ids`, `attention_mask` (dynamic batch size and sequence length)
- **Output**: `token_embeddings` (batch_size, sequence_length, 128) - L2 normalized per-token embeddings

### Usage with ONNX Runtime

```python
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{TARGET_REPO_ID}")
session = ort.InferenceSession("gte_moderncolbert.onnx")

text = "Your text here"
inputs = tokenizer(text, return_tensors="np")

outputs = session.run(
    ["token_embeddings"],
    {{"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}}
)
token_embeddings = outputs[0]  # Shape: (1, seq_len, 128)
```

"""

    # Append ONNX info to existing README or create new one
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            original_readme = f.read()
        with open(readme_path, "w") as f:
            f.write(original_readme + onnx_note)
        print("Updated README.md with ONNX information.")
    else:
        with open(readme_path, "w") as f:
            f.write(f"# {TARGET_REPO_ID}\n\n" + onnx_note)
        print("Created README.md with ONNX information.")

    # Upload the entire directory to the Hub
    print(f"Uploading to {TARGET_REPO_ID}...")
    api.upload_folder(
        folder_path=tmpdir,
        repo_id=TARGET_REPO_ID,
        commit_message=f"Add ONNX export of {MODEL_ID}",
    )

    print(f"✅ Successfully uploaded to https://huggingface.co/{TARGET_REPO_ID}")
