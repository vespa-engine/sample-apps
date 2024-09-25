from fasthtml.common import *
from importlib.util import find_spec

# Run find_spec for all the modules (imports will be removed by ruff if not used. This is just to check if the modules are available, and should be removed)ß
for module in ["torch", "einops", "PIL", "vidore_benchmark", "colpali_engine"]:
    spec = find_spec(module)
    assert spec is not None, f"Module {module} not found"

app, rt = fast_app()


@rt("/")
def get():
    return Div(P("Hello World!"), hx_get="/change")


serve()
