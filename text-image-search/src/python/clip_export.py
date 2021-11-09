#! /usr/bin/env python3

import os
import clip
import torch

class CLIPTransformerExporter(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text):
        return self.encode_text(text)

    def encode_text(self, text):
        x = self.model.token_embedding(text).type(self.model.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)

        # Need the cast to IntTensor for ORT support
        x = x[torch.arange(x.shape[0]), text.type(torch.IntTensor).argmax(dim=-1)] @ self.model.text_projection
        return x


device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
text = clip.tokenize(["dummy text to tokenize"]).to(device)
exporter = CLIPTransformerExporter(model)

os.makedirs("src/main/application/models", exist_ok=True)
torch.onnx.export(exporter, text,
                  "src/main/application/models/transformer.onnx",
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}})