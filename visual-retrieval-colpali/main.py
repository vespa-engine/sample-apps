import asyncio
import json

from fasthtml.common import *
from shad4fast import *
from vespa.application import Vespa

from backend.colpali import get_result_from_query, load_model
from backend.vespa_app import get_vespa_app
from frontend.app import Home, Search, SearchBox, SearchResult
from frontend.layout import Layout

highlight_js_theme_link = Link(id="highlight-theme", rel="stylesheet", href="")
highlight_js_theme = Script(src="/static/js/highlightjs-theme.js")
highlight_js = HighlightJS(
    langs=["python", "javascript", "java", "json", "xml"],
    dark="github-dark",
    light="github",
)

app, rt = fast_app(
    htmlkw={"cls": "h-full"},
    pico=False,
    hdrs=(
        ShadHead(tw_cdn=False, theme_handle=True),
        highlight_js,
        highlight_js_theme_link,
        highlight_js_theme,
    ),
)
vespa_app: Vespa = get_vespa_app()


class ModelManager:
    _instance = None
    model = None
    processor = None

    @staticmethod
    def get_instance():
        if ModelManager._instance is None:
            ModelManager._instance = ModelManager()
            ModelManager._instance.initialize_model_and_processor()
        return ModelManager._instance

    def initialize_model_and_processor(self):
        if self.model is None or self.processor is None:  # Ensure no reinitialization
            self.model, self.processor = load_model()
            if self.model is None or self.processor is None:
                print("Failed to initialize model or processor at startup")
            else:
                print("Model and processor loaded at startup")


@rt("/static/{filepath:path}")
def serve_static(filepath: str):
    return FileResponse(f"./static/{filepath}")


@rt("/")
def get():
    return Layout(Home())


@rt("/search")
def get(request):
    # Extract the 'query' and 'ranking' parameters from the URL
    query_value = request.query_params.get("query", "").strip()
    ranking_value = request.query_params.get("ranking", "option1")
    print("/search: Fetching results for ranking_value:", ranking_value)

    # Always render the SearchBox first
    if not query_value:
        # Show SearchBox and a message for missing query
        return Layout(
            Div(
                SearchBox(query_value=query_value, ranking_value=ranking_value),
                Div(
                    P(
                        "No query provided. Please enter a query.",
                        cls="text-center text-muted-foreground",
                    ),
                    cls="p-10",
                ),
                cls="grid",
            )
        )

    # Show the loading message if a query is provided
    return Layout(Search(request))  # Show SearchBox and Loading message initially


@rt("/fetch_results")
def get(request, query: str, nn: bool = True, sim_map: bool = True):
    # Check if the request came from HTMX; if not, redirect to /search
    if "hx-request" not in request.headers:
        return RedirectResponse("/search")

    # Extract ranking option from the request
    ranking_value = request.query_params.get("ranking", "option1")
    print(
        f"/fetch_results: Fetching results for query: {query}, ranking: {ranking_value}"
    )

    # Fetch model and processor
    manager = ModelManager.get_instance()
    model = manager.model
    processor = manager.processor

    # Fetch real search results from Vespa
    result = asyncio.run(
        get_result_from_query(
            vespa_app,
            processor=processor,
            model=model,
            query=query,
            nn=nn,
            gen_sim_map=sim_map,
        )
    )

    # Extract search results from the result payload
    search_results = (
        result["root"]["children"]
        if "root" in result and "children" in result["root"]
        else []
    )

    # Directly return the search results without the full page layout
    return SearchResult(search_results, show_sim_map=sim_map)


@rt("/app")
def get():
    return Layout(Div(P(f"Connected to Vespa at {vespa_app.url}"), cls="p-4"))


@rt("/run_query")
def get(query: str, nn: bool = False):
    # dummy-function to avoid running the query every time
    # result = get_result_dummy(query, nn)
    # If we want to run real, uncomment the following lines
    model, processor = get_model_and_processor()
    result = asyncio.run(
        get_result_from_query(
            vespa_app, processor=processor, model=model, query=query, nn=nn
        )
    )
    # model, processor = get_model_and_processor()
    # result = asyncio.run(
    #     get_result_from_query(vespa_app, processor=processor, model=model, query=query, nn=nn)
    # )
    return Layout(Div(H1("Result"), Pre(Code(json.dumps(result, indent=2))), cls="p-4"))


if __name__ == "__main__":
    # ModelManager.get_instance()  # Initialize once at startup
    serve(port=7860)
