"""Check app imports and startup after installing src/requirements.txt."""

import argparse
from contextlib import ExitStack
import importlib.metadata
import os
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))

    from dotenv import load_dotenv

    # Verify the Space SDK can be imported too.
    import gradio  # noqa: F401
    from starlette.testclient import TestClient
    import torch

    if args.env_file:
        load_dotenv(args.env_file.resolve())
    else:
        os.environ.update(
            {
                "USE_MTLS": "false",
                "VESPA_APP_TOKEN_URL": "https://example.invalid",
                "VESPA_CLOUD_SECRET_TOKEN": "test-token",
                "GEMINI_API_KEY": "test-key",
            }
        )

    with ExitStack() as stack:
        if not args.env_file:
            stack.enter_context(
                patch("vespa.application.Vespa.wait_for_application_up")
            )
        import main as app_module

        if not args.load_model:
            stack.enter_context(
                patch.object(
                    app_module, "SimMapGenerator", return_value=SimpleNamespace()
                )
            )
        with TestClient(app_module.app) as client:
            for route in ["/", "/about-this-demo", "/search"]:
                response = client.get(route)
                assert response.status_code == 200, route
            if args.load_model:
                # Mismatched transformers/colpali-engine versions can skip the
                # ColPali adapter weights, leaving the LoRA B matrices at zero.
                model = app_module.app.sim_map_generator.model
                lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
                assert lora_b and all(p.abs().sum() > 0 for p in lora_b), (
                    "ColPali adapter weights were not loaded"
                )
                embeddings, tokens = (
                    app_module.app.sim_map_generator.get_query_embeddings_and_token_map(
                        "return on investment"
                    )
                )
                assert embeddings.ndim == 2 and embeddings.shape[1] == 128
                assert torch.isfinite(embeddings).all() and tokens
    print("Hugging Face requirements install and app startup passed.")
    print("Gradio SDK:", importlib.metadata.version("gradio"))
    print("Real model loaded:", args.load_model)


if __name__ == "__main__":
    main()
