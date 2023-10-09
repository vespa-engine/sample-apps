# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import torch
import torch.onnx
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, emb_size, num_heads, hidden_dim_size, num_layers, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Embedding(vocab_size, emb_size)
        encoder_layers = TransformerEncoderLayer(emb_size, num_heads, hidden_dim_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src):
        src = self.encoder(src)
        output = self.transformer_encoder(src)
        return output


def main():
    vocabulary_size = 100
    embedding_size = 16
    hidden_dim_size = 32
    num_layers = 2
    num_heads = 2
    model = TransformerModel(vocabulary_size, embedding_size, num_heads, hidden_dim_size, num_layers)

    # Omit training - just export randomly initialized network

    sample_data = torch.IntTensor([[1,2,3,4,5]])
    torch.onnx.export(model,
                      sample_data,
                      "transformer.onnx",
                      input_names = ["input"],
                      output_names = ["output"],
                      dynamic_axes = {
                          "input": {0:"batch", 1:"tokens"},
                          "output": {0:"batch", 1:"tokens"},
                      },
                      opset_version=12)


if __name__ == "__main__":
    main()


