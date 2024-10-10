import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from fasthtml.common import *
from shad4fast import *
from vespa.application import Vespa

from backend.colpali import (
    load_model,
    get_result_from_query,
    get_query_embeddings_and_token_map,
    add_sim_maps_to_result,
)
from backend.vespa_app import get_vespa_app
from frontend.app import Home, Search, SearchBox, SearchResult
from frontend.layout import Layout
import hashlib
from collections import OrderedDict


# Initialize LRU Cache
class LRUCache:
    def __init__(self, max_size=20):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]


highlight_js_theme_link = Link(id="highlight-theme", rel="stylesheet", href="")
highlight_js_theme = Script(src="/static/js/highlightjs-theme.js")
highlight_js = HighlightJS(
    langs=["python", "javascript", "java", "json", "xml"],
    dark="github-dark",
    light="github",
)
sselink = Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")


app, rt = fast_app(
    htmlkw={"cls": "h-full"},
    pico=False,
    hdrs=(
        ShadHead(tw_cdn=False, theme_handle=True),
        highlight_js,
        highlight_js_theme_link,
        highlight_js_theme,
        sselink,
    ),
)
vespa_app: Vespa = get_vespa_app()

result_cache = LRUCache(max_size=20)
thread_pool = ThreadPoolExecutor()


def generate_query_id(query):
    return hashlib.md5(query.encode("utf-8")).hexdigest()


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
async def get(request, query: str, nn: bool = True):
    if "hx-request" not in request.headers:
        return RedirectResponse("/search")

    ranking_value = request.query_params.get("ranking", "option1")
    print(
        f"/fetch_results: Fetching results for query: {query}, ranking: {ranking_value}"
    )
    # Generate a unique query_id based on the query and ranking value
    query_id = generate_query_id(query + ranking_value)

    if "bm25" in ranking_value:
        nn = False

    # Fetch model and processor
    manager = ModelManager.get_instance()
    model = manager.model
    processor = manager.processor
    q_embs, token_to_idx = get_query_embeddings_and_token_map(processor, model, query)

    # Fetch real search results from Vespa
    result = await get_result_from_query(
        app=vespa_app,
        processor=processor,
        model=model,
        query=query,
        q_embs=q_embs,
        token_to_idx=token_to_idx,
        nn=nn,
    )
    # Start generating the similarity map in the background
    asyncio.create_task(
        generate_similarity_map(
            model, processor, query, q_embs, token_to_idx, result, query_id
        )
    )
    print("Search results fetched")
    search_results = (
        result["root"]["children"]
        if "root" in result and "children" in result["root"]
        else []
    )
    return SearchResult(search_results, query_id)


async def generate_similarity_map(
    model, processor, query, q_embs, token_to_idx, result, query_id
):
    loop = asyncio.get_event_loop()
    sim_map_task = partial(
        add_sim_maps_to_result,
        result=result,
        model=model,
        processor=processor,
        query=query,
        q_embs=q_embs,
        token_to_idx=token_to_idx,
    )
    sim_map_result = await loop.run_in_executor(thread_pool, sim_map_task)
    result_cache.set(query_id, sim_map_result)


@app.get("/updated_search_results")
async def updated_search_results(query_id: str):
    data = result_cache.get(query_id)
    if data is None:
        return HTMLResponse(status_code=204)
    search_results = (
        data["root"]["children"]
        if "root" in data and "children" in data["root"]
        else []
    )
    updated_content = SearchResult(results=search_results, query_id=None)
    return updated_content


@rt("/app")
def get():
    return Layout(Div(P(f"Connected to Vespa at {vespa_app.url}"), cls="p-4"))


if __name__ == "__main__":
    # ModelManager.get_instance()  # Initialize once at startup
    serve(port=7860)
