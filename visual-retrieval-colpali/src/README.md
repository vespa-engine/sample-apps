---
title: ColPali 🤝 Vespa - Visual Retrieval
short_description: Visual Retrieval with ColPali and Vespa
emoji: 👀
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: main.py
pinned: false
license: apache-2.0
suggested_hardware: t4-small
models:
  - vidore/colpaligemma-3b-pt-448-base
  - vidore/colpali-v1.2
preload_from_hub:
  - vidore/colpaligemma-3b-pt-448-base config.json,model-00001-of-00002.safetensors,model-00002-of-00002.safetensors,model.safetensors.index.json,preprocessor_config.json,special_tokens_map.json,tokenizer.json,tokenizer_config.json 12c59eb7e23bc4c26876f7be7c17760d5d3a1ffa
  - vidore/colpali-v1.2 adapter_config.json,adapter_model.safetensors,preprocessor_config.json,special_tokens_map.json,tokenizer.json,tokenizer_config.json 9912ce6f8a462d8cf2269f5606eabbd2784e764f
---