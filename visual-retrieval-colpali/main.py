import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from fasthtml.common import *
from shad4fast import *
from vespa.application import Vespa
import time

from backend.colpali import (
    get_result_from_query,
    get_query_embeddings_and_token_map,
    add_sim_maps_to_result,
    is_special_token,
)
from backend.vespa_app import get_vespa_app
from backend.cache import LRUCache
from backend.modelmanager import ModelManager
from frontend.app import (
    Home,
    Search,
    SearchBox,
    SearchResult,
    SimMapButtonPoll,
    SimMapButtonReady,
)
from frontend.layout import Layout
import hashlib

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

result_cache = LRUCache(max_size=20)  # Each result can be ~10MB
task_cache = LRUCache(
    max_size=1000
)  # Map from query_id to boolean value - False if not all results are ready.
thread_pool = ThreadPoolExecutor()


@app.on_event("startup")
def load_model_on_startup():
    app.manager = ModelManager.get_instance()
    return


def generate_query_id(query):
    return hashlib.md5(query.encode("utf-8")).hexdigest()


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
    ranking_value = request.query_params.get("ranking", "nn+colpali")
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

    # Extract ranking option from the request
    ranking_value = request.query_params.get("ranking")
    print(
        f"/fetch_results: Fetching results for query: {query}, ranking: {ranking_value}"
    )
    # Generate a unique query_id based on the query and ranking value
    query_id = generate_query_id(query + ranking_value)
    # See if results are already in cache
    if result_cache.get(query_id):
        print(f"Results for query_id {query_id} already in cache")
        result = result_cache.get(query_id)
        search_results = get_results_children(result)
        # If task is completed, return the results, but no query_id
        if task_cache.get(query_id):
            return SearchResult(search_results, None)
        # If task is not completed, return the results with query_id
        return SearchResult(search_results, query_id)
    task_cache.set(query_id, False)
    model = app.manager.model
    processor = app.manager.processor
    q_embs, token_to_idx = get_query_embeddings_and_token_map(processor, model, query)

    start = time.perf_counter()
    # Fetch real search results from Vespa
    result = await get_result_from_query(
        app=vespa_app,
        processor=processor,
        model=model,
        query=query,
        q_embs=q_embs,
        token_to_idx=token_to_idx,
        ranking=ranking_value,
    )
    end = time.perf_counter()
    print(
        f"Search results fetched in {end - start:.2f} seconds, Vespa says searchtime was {result['timing']['searchtime']} seconds"
    )
    # Start generating the similarity map in the background
    asyncio.create_task(
        generate_similarity_map(
            model, processor, query, q_embs, token_to_idx, result, query_id
        )
    )
    fields_to_add = [
        f"sim_map_{token}"
        for token in token_to_idx.keys()
        if not is_special_token(token)
    ]
    search_results = get_results_children(result)
    for result in search_results:
        for sim_map_key in fields_to_add:
            result["fields"][sim_map_key] = None
    return SearchResult(search_results, query_id)


def get_results_children(result):
    search_results = (
        result["root"]["children"]
        if "root" in result and "children" in result["root"]
        else []
    )
    return search_results


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
        query_id=query_id,
        result_cache=result_cache,
    )
    sim_map_result = await loop.run_in_executor(thread_pool, sim_map_task)
    result_cache.set(query_id, sim_map_result)
    task_cache.set(query_id, True)


@app.get("/get_sim_map")
async def get_sim_map(query_id: str, idx: int, token: str):
    """
    Endpoint that each of the sim map button polls to get the sim map image
    when it is ready. If it is not ready, returns a SimMapButtonPoll, that
    continues to poll every 1 second.
    """
    result = result_cache.get(query_id)
    if result is None:
        return SimMapButtonPoll(query_id=query_id, idx=idx, token=token)
    search_results = get_results_children(result)
    # Check if idx exists in list of children
    if idx >= len(search_results):
        return SimMapButtonPoll(query_id=query_id, idx=idx, token=token)
    else:
        sim_map_key = f"sim_map_{token}"
        sim_map_b64 = search_results[idx]["fields"].get(sim_map_key, None)
        if sim_map_b64 is None:
            return SimMapButtonPoll(query_id=query_id, idx=idx, token=token)
        sim_map_img_src = f"data:image/jpeg;base64,{sim_map_b64}"
        return SimMapButtonReady(
            query_id=query_id, idx=idx, token=token, img_src=sim_map_img_src
        )


@rt("/app")
def get():
    return Layout(Div(P(f"Connected to Vespa at {vespa_app.url}"), cls="p-4"))


if __name__ == "__main__":
    # ModelManager.get_instance()  # Initialize once at startup
    serve(port=7860)
